from routefinder import PathFinder
from predicted_volumes_gru import predicted_volumes  # Change to lstm or svr as needed
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <mapfile> <search_method>")
        print("Example: python main.py map.txt as")
        sys.exit(1)

    mapfile = sys.argv[1]
    method = sys.argv[2]

    finder = PathFinder(mapfile)
    finder.apply_predicted_volumes(predicted_volumes)
    goal, nodes_created, path = finder.run_search(method)

    if goal:
        print(f"{mapfile} {method}")
        print(f"{goal} {nodes_created}")
        print(" ".join(map(str, path)))
    else:
        print(f"{mapfile} {method}")
        print("No solution found")

if __name__ == "__main__":
    main()
