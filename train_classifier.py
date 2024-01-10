import argparse
import os
import joblib


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a COICOP classifier')
    parser.add_argument("-i", "--input-filename", type=str,
                        required=True, help="The input filename in parquet format")
    parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text",
                        help="The column name containing the receipt text")
    parser.add_argument("-c", "--coicop-column", type=str, default="coicop",
                        help="The column name containing the COICOP code")
    parser.add_argument("-f", "--feature-extractor", type=str,
                        required=True, help="The feature extractor to use")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="The model to use")
    parser.add_argument("-o", "--output-filename", type=str,
                        required=True, help="The output filename for the pipeline")
    args = parser.parse_args()
    main(args)
