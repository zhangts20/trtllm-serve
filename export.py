import sys

from tllm.args import get_args
from tllm.build_helper import export


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python main.py test.yml")
        exit(1)
    args = get_args(sys.argv[1])

    export(args)
