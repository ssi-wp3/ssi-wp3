from typing import Any, Dict, List
import argparse
import os
import joblib
import numpy as np

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


def main(args):
    pipeline = CoicopPipeline(args.pipeline_path)
    predictions = pipeline.predict_proba(args.inputs)
    print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict COICOP')
    parser.add_argument("-pp", "--pipeline-path", type=str,
                        default="pipeline.joblib", help="Path to pipeline")
    parser.add_argument("-i", "--input_data", type=str,
                        required=True, help="Path to the input json file")
    parser.add_argument("-o", "--output_data", type=str,
                        required=True, help="Path to the output json file")
    parser.add_argument("-p", "--params", type=str,
                        required=True, help="Path to the params json file")
    args = parser.parse_args()
    main(args)
