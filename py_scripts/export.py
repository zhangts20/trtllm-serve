import sys
from args import get_args
from build_helper import export


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py export.yml")
        exit(1)
    args = get_args(sys.argv[1])

    export(args)
