import logging
import argparse


def main(args):
    pass


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv")
    args = parser.parse_args()

    main(args)
