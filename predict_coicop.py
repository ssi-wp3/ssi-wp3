from ssi.coicop_json_parser import CoicopInputFile, CoicopOutputFile, ProductClassificationResult, CoicopClassification, load_input_file
from typing import Any, Dict, List
import argparse
import joblib
import json
# Adapted from https://huggingface.co/scikit-learn/sklearn-transformers/blob/main/pipeline.py


class CoicopPipeline():
    def __init__(self, pipeline_path: str):
        # load the model
        self.model = joblib.load(pipeline_path)

    def __call__(self, inputs: List[str]):
        return self.predict_proba(inputs)

    def predict(self, inputs: List[str]) -> List[str]:
        return self.model.predict(inputs)

    def predict_proba(self, inputs: List[str]) -> List[Dict[str, Any]]:
        predictions = self.model.predict_proba(inputs)
        prediction_labels = []
        for prediction in predictions:
            label_predictions = dict()
            for index, probability in enumerate(prediction):
                label = self.model.classes_[index]
                label_predictions[label] = probability,
            prediction_labels.append(label_predictions)
        return prediction_labels

    def predict_receipt(self, receipt_input: CoicopInputFile) -> CoicopOutputFile:
        classification_output = CoicopOutputFile(coicop_classification_request=receipt_input.coicop_classification_request,
                                                 receipt=receipt_input.receipt
                                                 )

        receipt_ids = [item.id for item in receipt_input.receipt.items]
        receipt_descriptions = [item.description
                                for item in receipt_input.receipt.items]
        predicted_probabilities = self.predict_proba(receipt_descriptions)

        for receipt_id, probabilities in zip(receipt_ids, predicted_probabilities):
            coicop_codes = [
                CoicopClassification(code=coicop_code, confidence=probability)
                for coicop_code, probability in probabilities.items()
            ]
            classification_result = ProductClassificationResult(
                id=receipt_id, coicop_codes=coicop_codes)
            classification_output.coicop_classification_result.result.append(
                classification_result)

        return classification_output


def main(args):
    try:
        pipeline = CoicopPipeline(args.pipeline_path)
        coicop_input_file = load_input_file(args.input_data)
        coicop_output_file = pipeline.predict_receipt(coicop_input_file)

        with open(args.output_data, "w") as json_file:
            output_json = coicop_output_file.model_dump()
            json.dump(output_json, json_file, indent=4)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict COICOP')
    parser.add_argument("-pp", "--pipeline-path", type=str,
                        required=True, help="Path to pipeline")
    parser.add_argument("-i", "--input_data", type=str,
                        required=True, help="Path to the input json file")
    parser.add_argument("-o", "--output_data", type=str,
                        required=True, help="Path to the output json file")
    parser.add_argument("-p", "--params", type=str,
                        default=None, help="Path to the params json file (optional)")
    args = parser.parse_args()
    main(args)
