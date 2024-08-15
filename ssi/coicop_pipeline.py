from ssi.coicop_json_parser import CoicopInputFile, CoicopOutputFile,  create_coicop_output_file
from typing import Any, Dict, List
from enum import Enum
from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import joblib


class CoicopPipelineType(Enum):
    sklearn = 1
    hugging_face = 2


class Pipeline(ABC):
    @abstractmethod
    def load(self, pipeline_path: str):
        pass

    @abstractmethod
    def predict(self, inputs: List[str]) -> List[str]:
        pass

    @abstractmethod
    def predict_proba(self, inputs: List[str]) -> List[Dict[str, Any]]:
        pass


class SklearnPipeline(Pipeline):
    def __init__(self):
        self.model = None

    def load(self, pipeline_path: str):
        self.model = joblib.load(pipeline_path)

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


class HuggingFacePipeline(Pipeline):
    def __init__(self):
        self.model = None

    def load(self, pipeline_path: str):
        # TODO check whether COICOP labels are loaded when we save them in the training script
        tokenizer = AutoTokenizer.from_pretrained(pipeline_path)
        classification_model = AutoModelForSequenceClassification.from_pretrained(
            pipeline_path)
        self.model = TextClassificationPipeline(
            model=classification_model, tokenizer=tokenizer, top_k=None)

    def predict(self, inputs: List[str]) -> List[str]:
        return self.model(inputs, return_all_scores=False)

    def predict_proba(self, inputs: List[str]) -> List[Dict[str, Any]]:
        return self.model(inputs, return_all_scores=True)


class PipelineFactory:
    @staticmethod
    def pipeline_for(pipeline_type: CoicopPipelineType) -> Pipeline:
        if pipeline_type == CoicopPipelineType.sklearn:
            return SklearnPipeline()
        elif pipeline_type == CoicopPipelineType.hugging_face:
            return HuggingFacePipeline()
        else:
            raise ValueError("Pipeline type not supported")


class CoicopPipeline:
    def __init__(self, pipeline_path: str, pipeline_type: CoicopPipelineType):
        # load the model
        self.model = PipelineFactory.pipeline_for(
            pipeline_type).load(pipeline_path)
        print("Model in init", self.model)

    def __call__(self, inputs: List[str]):
        return self.model.predict_proba(inputs)

    def predict(self, inputs: List[str]) -> List[str]:
        return self.model.predict(inputs)

    def predict_proba(self, inputs: List[str]) -> List[Dict[str, Any]]:
        print(self.model)
        return self.model.predict_proba(inputs)

    def predict_receipt(self, receipt_input: CoicopInputFile) -> CoicopOutputFile:
        receipt_ids = [item.id for item in receipt_input.receipt.items]
        receipt_descriptions = [item.description
                                for item in receipt_input.receipt.items]
        predicted_probabilities = self.predict_proba(receipt_descriptions)

        return create_coicop_output_file(receipt_input, receipt_ids, predicted_probabilities)
