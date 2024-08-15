from ssi.coicop_json_parser import load_input_file
from ssi.coicop_pipeline import CoicopPipeline, CoicopPipelineType
import argparse
import json
import pandas as pd
# Adapted from https://huggingface.co/scikit-learn/sklearn-transformers/blob/main/pipeline.py


def main(args):
    try:
        pipeline = CoicopPipeline(args.pipeline_path, args.pipeline_type)
        coicop_input_file = load_input_file(args.input_data)

        coicop_mapping = None
        if args.coicop_code_list is not None:
            coicop_mapping = pd.read_csv(
                args.coicop_code_list, delimiter=args.delimiter, index_col=0, dtype={args.coicop_column: str, args.coicop_description_column: str})
        coicop_output_file = pipeline.predict_receipt(
            coicop_input_file, coicop_mapping, args.coicop_description_column)

        with open(args.output_data, "w") as json_file:
            output_json = coicop_output_file.model_dump()
            json.dump(output_json, json_file, indent=4)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict COICOP')
    parser.add_argument("-pp", "--pipeline-path", type=str,
                        required=True, help="Path to pipeline")
    parser.add_argument("-pt", "--pipeline-type", type=CoicopPipelineType, choices=list(CoicopPipelineType),
                        default=CoicopPipelineType.hugging_face, help="Type of pipeline to use for prediction")
    parser.add_argument("-i", "--input_data", type=str,
                        required=True, help="Path to the input json file")
    parser.add_argument("-o", "--output_data", type=str,
                        required=True, help="Path to the output json file")
    parser.add_argument("-c", "--coicop-code-list", type=str,
                        default=None, help="Path to the COICOP code list")
    parser.add_argument("-d", "--delimiter", type=str,
                        default=";", help="Delimiter for the COICOP code list")
    parser.add_argument("-cc", "--coicop-column", type=str,
                        default="coicop_number", help="Column name for the COICOP code")
    parser.add_argument("-cn", "--coicop-description-column", type=str,
                        default="coicop_name", help="Column name for the COICOP name")
    parser.add_argument("-p", "--params", type=str,
                        default=None, help="Path to the params json file (optional)")
    args = parser.parse_args()
    main(args)
