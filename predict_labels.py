from ssi.coicop_pipeline import CoicopPipeline
from ssi.predict import predict_from_file
import argparse
import json
# Adapted from https://huggingface.co/scikit-learn/sklearn-transformers/blob/main/pipeline.py

def main(args):
    try:
        pipeline = CoicopPipeline(args.pipeline_path)
        predict_from_file(pipeline, args.input_data, args.output_data, args.receipt_text_column, args.label_column)
       
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict COICOP')
    parser.add_argument("-pp", "--pipeline-path", type=str,
                        required=True, help="Path to pipeline")
    parser.add_argument("-i", "--input-data", type=str,
                        required=True, help="Path to the input parquet file")
    parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text", help="The column name containing the receipt text")
    parser.add_argument("-l", "--label-column", type=str, default="coicop_number", help="The column name containing the COICOP code")
    parser.add_argument("-o", "--output-data", type=str,
                        required=True, help="Path to the output parquet file")
    args = parser.parse_args()
    main(args)