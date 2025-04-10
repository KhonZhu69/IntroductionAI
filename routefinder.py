#!/usr/bin/env python3
# ^ This shebang line tells the system to use Python 3 when executing this file directly

import heapq
import math
import sys
from collections import deque


# Node class represents a state in our search problem
# Each node knows its state (node ID), parent node, and path cost from start
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state  # node number
        self.parent = parent
        self.action = action
        self.path_cost = path_cost  # Cost from start to here
        self.depth = 0 if parent is None else parent.depth + 1  # Depth in the search tree

    # This helps priority queue know how to order nodes when costs are equal
    def __lt__(self, other):
        # For priority queue ordering when costs are equal, node number as tiebreaker
        return self.state < other.state


# PriorityQueue to help with informed search algorithms like A* and GBFS
class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.count = 0

    def empty(self):
        return len(self.elements) == 0

    # Add an item with its priority to the queue
    def put(self, item, priority):
        # Use count to break ties consistently when priorities are equal
        heapq.heappush(self.elements, (priority, self.count, item))
        self.count += 1

    # Get the item with the lowest priority
    def get(self):
        return heapq.heappop(self.elements)[2]


# Main class that handles the path finding algorithms
class PathFinder:
    def __init__(self, filename):
        self.nodes = {}  # Node ID -> (x, y) coordinates
        self.graph = {}  # Node ID -> [(neighbor, cost), ...]
        self.origin = None
        self.destinations = []
        self.nodes_created = 0  # Track nodes created during search (for efficiency comparison)

        self.parse_file(filename)

    # Parse the input file to build the graph
    def parse_file(self, filename):
        with open(filename, 'r') as file:
            content = file.read()

        sections = content.split('\n\n')

        # If the file doesn't have empty lines between sections, try alternative parsing
        if len(sections) < 4:
            content_lines = content.split('\n')
            nodes_section = []
            edges_section = []
            origin_section = []
            dest_section = []

            current_section = None

            # Determine which section each line belongs to
            for line in content_lines:
                if line.strip() == "Nodes:":
                    current_section = "nodes"
                    continue
                elif line.strip() == "Edges:":
                    current_section = "edges"
                    continue
                elif line.strip() == "Origin:":
                    current_section = "origin"
                    continue
                elif line.strip() == "Destinations:":
                    current_section = "dest"
                    continue

                if current_section == "nodes":
                    nodes_section.append(line)
                elif current_section == "edges":
                    edges_section.append(line)
                elif current_section == "origin":
                    origin_section.append(line)
                elif current_section == "dest":
                    dest_section.append(line)
        else:
            # Standard parsing for well-formatted files with blank lines
            nodes_section = sections[0].split('\n')[1:]  # Skip "Nodes:" line
            edges_section = sections[1].split('\n')[1:]  # Skip "Edges:" line
            origin_section = [sections[2].split('\n')[1]]  # Skip "Origin:" line
            dest_section = [sections[3].split('\n')[1]]  # Skip "Destinations:" line

        # Process each node (ID and coordinates)
        for line in nodes_section:
            if not line.strip():
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue
            node_id = parts[0].strip()
            coords = parts[1].strip()
            try:
                node_id = int(node_id)
                x, y = map(int, coords.strip(' ()').split(','))
                self.nodes[node_id] = (x, y)
                self.graph[node_id] = []
            except ValueError:
                continue  # Skip if conversion fails

        # Process each edge between nodes
        for line in edges_section:
            if not line.strip():
                continue
            parts = line.split(':')
            if len(parts) != 2:
                continue
            edge = parts[0].strip()
            cost = parts[1].strip()
            try:
                edge = edge.replace('(', '').replace(')', '')
                from_node, to_node = map(int, edge.split(','))
                cost = int(cost)
                self.graph[from_node].append((to_node, cost))
            except (ValueError, KeyError):
                continue  # Skip if conversion fails

        # Find the starting node (origin)
        for line in origin_section:
            if line.strip():
                try:
                    self.origin = int(line.strip())
                    break
                except ValueError:
                    continue

        # Find goal nodes (destinations)
        for line in dest_section:
            if line.strip():
                try:
                    self.destinations = [int(d.strip()) for d in line.split(';')]
                    break
                except ValueError:
                    continue

    # Reset the node counter for a new search
    def reset_counter(self):
        self.nodes_created = 0

    # Calculate the straight-line distance between two nodes (Euclidean distance)
    def heuristic(self, node1, node2):
        """Euclidean distance heuristic"""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Get the minimum distance to any destination (used as heuristic)
    def min_heuristic(self, node):
        """Minimum heuristic to any destination"""
        return min(self.heuristic(node, dest) for dest in self.destinations)

    # Reconstruct the path from start to goal
    def get_solution(self, node):
        """Extract the path from start to goal"""
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        path.reverse()
        return path

    # Depth-First Search algorithm (explores deeply before backtracking)
    def dfs(self):
        """Depth-First Search"""
        self.reset_counter()

        start_node = Node(state=self.origin)
        self.nodes_created += 1

        # If the origin is already a destination, we're done
        if self.origin in self.destinations:
            return self.origin, self.nodes_created, [self.origin]

        frontier = [start_node]
        explored = set()

        while frontier:
            # DFS uses a stack (LIFO) - get the last added node
            node = frontier.pop()

            # Check if we've reached a destination
            if node.state in self.destinations:
                return node.state, self.nodes_created, self.get_solution(node)

            explored.add(node.state)

            # Get neighbors sorted by node ID (ascending)
            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[node.state]],
                               key=lambda x: x[0])

            # Add neighbors to frontier in reverse order (so smallest ID is popped first)
            for neighbor, cost in reversed(neighbors):
                if neighbor not in explored and not any(n.state == neighbor for n in frontier):
                    child = Node(state=neighbor, parent=node, path_cost=node.path_cost + cost)
                    self.nodes_created += 1
                    frontier.append(child)

        return None, self.nodes_created, []

    # Breadth-First Search algorithm (explores level by level)
    def bfs(self):
        """Breadth-First Search"""
        self.reset_counter()

        start_node = Node(state=self.origin)
        self.nodes_created += 1

        if self.origin in self.destinations:
            return self.origin, self.nodes_created, [self.origin]

        # BFS uses a queue (FIFO)
        frontier = deque([start_node])
        explored = set()
        frontier_states = {self.origin}

        while frontier:
            # Get the first node added to the queue
            node = frontier.popleft()
            frontier_states.remove(node.state)

            if node.state in self.destinations:
                return node.state, self.nodes_created, self.get_solution(node)

            explored.add(node.state)

            # Get neighbors sorted by node ID (ascending)
            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[node.state]],
                               key=lambda x: x[0])

            for neighbor, cost in neighbors:
                if neighbor not in explored and neighbor not in frontier_states:
                    child = Node(state=neighbor, parent=node, path_cost=node.path_cost + cost)
                    self.nodes_created += 1
                    frontier.append(child)
                    frontier_states.add(neighbor)

        return None, self.nodes_created, []

    # Greedy Best-First Search (always expands the node closest to the goal)
    def gbfs(self):
        """Greedy Best-First Search"""
        self.reset_counter()

        start_node = Node(state=self.origin)
        self.nodes_created += 1

        if self.origin in self.destinations:
            return self.origin, self.nodes_created, [self.origin]

        # GBFS uses a priority queue ordered by heuristic values
        frontier = PriorityQueue()
        frontier.put(start_node, self.min_heuristic(self.origin))
        explored = set()
        frontier_states = {self.origin}

        while not frontier.empty():
            node = frontier.get()
            frontier_states.remove(node.state)

            if node.state in self.destinations:
                return node.state, self.nodes_created, self.get_solution(node)

            explored.add(node.state)

            # Get neighbors sorted by node ID (for tiebreaking purposes)
            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[node.state]],
                               key=lambda x: x[0])

            for neighbor, cost in neighbors:
                if neighbor not in explored and neighbor not in frontier_states:
                    child = Node(state=neighbor, parent=node, path_cost=node.path_cost + cost)
                    self.nodes_created += 1
                    # Use only the heuristic to prioritize (how close to goal)
                    frontier.put(child, self.min_heuristic(neighbor))
                    frontier_states.add(neighbor)

        return None, self.nodes_created, []

    # A* Search (combines path cost and heuristic for optimal paths)
    def astar(self):
        """A* Search"""
        self.reset_counter()

        start_node = Node(state=self.origin)
        self.nodes_created += 1

        if self.origin in self.destinations:
            return self.origin, self.nodes_created, [self.origin]

        frontier = PriorityQueue()
        # A* uses f(n) = g(n) + h(n) - path cost so far plus heuristic
        frontier.put(start_node, 0 + self.min_heuristic(self.origin))  # f = g + h
        explored = set()
        frontier_states = {self.origin: 0}  # state -> path_cost

        while not frontier.empty():
            node = frontier.get()

            if node.state in self.destinations:
                return node.state, self.nodes_created, self.get_solution(node)

            explored.add(node.state)
            frontier_states.pop(node.state, None)

            # Get neighbors sorted by node ID (for tiebreaking purposes)
            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[node.state]],
                               key=lambda x: x[0])

            for neighbor, cost in neighbors:
                path_cost = node.path_cost + cost

                if neighbor in explored:
                    continue

                # If we find a better path to a neighbor, update it
                if neighbor not in frontier_states or path_cost < frontier_states[neighbor]:
                    child = Node(state=neighbor, parent=node, path_cost=path_cost)
                    self.nodes_created += 1
                    f_cost = path_cost + self.min_heuristic(neighbor)
                    frontier.put(child, f_cost)
                    frontier_states[neighbor] = path_cost

        return None, self.nodes_created, []

    # Uniform Cost Search (expands nodes by path cost - like Dijkstra's algorithm)
    def ucs(self):
        """Uniform Cost Search (Custom Uninformed Strategy #1)"""
        self.reset_counter()

        start_node = Node(state=self.origin)
        self.nodes_created += 1

        if self.origin in self.destinations:
            return self.origin, self.nodes_created, [self.origin]

        frontier = PriorityQueue()
        # UCS prioritizes by path cost only (no heuristic)
        frontier.put(start_node, 0)
        explored = set()
        frontier_states = {self.origin: 0}  # state -> path_cost

        while not frontier.empty():
            node = frontier.get()

            if node.state in self.destinations:
                return node.state, self.nodes_created, self.get_solution(node)

            explored.add(node.state)
            frontier_states.pop(node.state, None)

            # Get neighbors sorted by node ID (for tiebreaking purposes)
            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[node.state]],
                               key=lambda x: x[0])

            for neighbor, cost in neighbors:
                path_cost = node.path_cost + cost

                if neighbor in explored:
                    continue

                if neighbor not in frontier_states or path_cost < frontier_states[neighbor]:
                    child = Node(state=neighbor, parent=node, path_cost=path_cost)
                    self.nodes_created += 1
                    frontier.put(child, path_cost)
                    frontier_states[neighbor] = path_cost

        return None, self.nodes_created, []

    # Bidirectional Search (searches from both start and goal simultaneously)
    def bidirectional_search(self):
        """Bidirectional Search (Custom Informed Strategy #2)
        This implementation works when there's a single destination.
        If multiple destinations, it uses the first one.
        """
        self.reset_counter()
        destination = self.destinations[0]  # Use first destination

        if self.origin == destination:
            return self.origin, self.nodes_created, [self.origin]

        # Forward search from origin
        forward_frontier = {self.origin: Node(state=self.origin)}
        forward_explored = {}

        # Backward search from destination
        backward_frontier = {destination: Node(state=destination)}
        backward_explored = {}

        self.nodes_created += 2  # Count start and goal nodes

        while forward_frontier and backward_frontier:
            # Check if forward and backward searches have met
            intersection = set(forward_frontier.keys()) & set(backward_frontier.keys())
            if intersection:
                meeting_point = min(intersection)  # Use smallest node ID if multiple intersections
                # Construct path from origin to meeting point
                forward_path = self.get_solution(forward_frontier[meeting_point])
                # Construct path from destination to meeting point
                backward_path = self.get_solution(backward_frontier[meeting_point])
                backward_path.reverse()
                backward_path.pop(0)  # Remove duplicated meeting point
                # Combine paths
                path = forward_path + backward_path
                return destination, self.nodes_created, path

            # Expand forward (from origin)
            current = min(forward_frontier.keys())  # Use smallest node ID
            current_node = forward_frontier.pop(current)
            forward_explored[current] = current_node

            neighbors = sorted([(neighbor, cost) for neighbor, cost in self.graph[current]],
                               key=lambda x: x[0])

            for neighbor, cost in neighbors:
                if neighbor in forward_explored:
                    continue

                if neighbor not in forward_frontier:
                    child = Node(state=neighbor, parent=current_node, path_cost=current_node.path_cost + cost)
                    self.nodes_created += 1
                    forward_frontier[neighbor] = child

            # Expand backward (from destination)
            if not backward_frontier:
                break

            current = min(backward_frontier.keys())  # Use smallest node ID
            current_node = backward_frontier.pop(current)
            backward_explored[current] = current_node

            # Finding incoming edges is trickier - we need to check all nodes
            backward_neighbors = []
            for node in self.graph:
                for neighbor, cost in self.graph[node]:
                    if neighbor == current:
                        backward_neighbors.append((node, cost))

            backward_neighbors.sort()  # Sort by node ID

            for neighbor, cost in backward_neighbors:
                if neighbor in backward_explored:
                    continue

                if neighbor not in backward_frontier:
                    child = Node(state=neighbor, parent=current_node, path_cost=current_node.path_cost + cost)
                    self.nodes_created += 1
                    backward_frontier[neighbor] = child

        return None, self.nodes_created, []

    # Run the selected search method based on the command line argument
    def run_search(self, method):
        """Run a search with the specified method"""
        method = method.lower()

        if method == "dfs":
            return self.dfs()
        elif method == "bfs":
            return self.bfs()
        elif method == "gbfs":
            return self.gbfs()
        elif method == "as":
            return self.astar()
        elif method == "cus1":
            return self.ucs()  # Uniform Cost Search (custom uninformed)
        elif method == "cus2":
            return self.bidirectional_search()  # Bidirectional Search (custom informed)
        else:
            print(f"Unknown method: {method}")
            return None, 0, []


# Check the Python version - helpful for debugging
def check_python_version():
    version = sys.version_info
    print(f"Running Python version: {version.major}.{version.minor}.{version.micro}")


# Main function to handle command line arguments and run the program
def main():
    # Uncomment the line below if you want to see which Python version you're using
    # check_python_version()

    if len(sys.argv) != 3:
        print("Usage: python routefinder.py <filename> <method>")
        print("   or: python3 routefinder.py <filename> <method>")
        print("   or: ./routefinder.py <filename> <method> (after running chmod +x routefinder.py)")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2]

    path_finder = PathFinder(filename)
    goal, nodes_created, path = path_finder.run_search(method)

    # Print output in the required format
    if goal is not None:
        print(f"{filename} {method}")
        print(f"{goal} {nodes_created}")
        print(" ".join(map(str, path)))
    else:
        print(f"{filename} {method}")
        print("No solution found")


if __name__ == "__main__":
    main()