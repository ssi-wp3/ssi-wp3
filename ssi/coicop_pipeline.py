from ssi.coicop_json_parser import CoicopInputFile, CoicopOutputFile,  create_coicop_output_file
from typing import Any, Dict, List
import joblib


class CoicopPipeline:
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
                label_predictions[label] = probability
            prediction_labels.append(label_predictions)
        return prediction_labels

    def predict_receipt(self, receipt_input: CoicopInputFile) -> CoicopOutputFile:
        receipt_ids = [item.id for item in receipt_input.receipt.items]
        receipt_descriptions = [item.description
                                for item in receipt_input.receipt.items]
        predicted_probabilities = self.predict_proba(receipt_descriptions)
        
        return create_coicop_output_file(receipt_input, receipt_ids, predicted_probabilities)