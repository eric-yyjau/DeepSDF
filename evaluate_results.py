
import argparse
import logging
import json
import numpy as np
import os


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--file",
        "-f",
        dest="file",
        required=True,
        help="csv file with evaluate results",
    )

    args = arg_parser.parse_args()
    
    result = np.genfromtxt(args.file, delimiter=',', dtype=None)
    def print_result(arr):
        print(f"mean: {arr.mean()}")
        print(f"median: {np.median(arr)}")
        pass

    print(f"result: {result}")
    print_result(result[1:,1].astype(np.float))