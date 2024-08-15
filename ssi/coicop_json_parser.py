
from pydantic import BaseModel, field_serializer
from datetime import date
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


class ReceiptItem(BaseModel):
    id: str
    description: str
    quantity: Optional[int]
    unit_price: Optional[float]
    total_price: Optional[float]


class Receipt(BaseModel):
    store: Optional[str]
    date: Optional[date]
    items: List[ReceiptItem]
    total: Optional[float]
    currency: Optional[str]
    language_hint: Optional[str]
    metadata: Optional[dict]

    @field_serializer('date')
    def serialize_date(self, date: date, _info):
        return date.strftime('%Y-%m-%d')


class CoicopInputFile(BaseModel):
    coicop_classification_request: List[str]
    receipt: Receipt


class CoicopClassification(BaseModel):
    code: str
    description: Optional[str]
    confidence: Optional[float]


class ProductClassificationResult(BaseModel):
    id: str
    coicop_codes: List[CoicopClassification]


class ClassificationResult(BaseModel):
    result: List[ProductClassificationResult]


class CoicopOutputFile(CoicopInputFile):
    coicop_classification_result: ClassificationResult
    metadata: Optional[dict]


def load_input_file(filename: str) -> CoicopInputFile:
    with open(filename, "r") as json_file:
        return CoicopInputFile.model_validate_json(json_file.read())


def get_description(coicop_code: str, coicop_mapping: Optional[pd.DataFrame]) -> str:
    if coicop_mapping is None:
        return ""
    return coicop_mapping[coicop_code]


def create_coicop_output_file(receipt_input: CoicopInputFile,
                              receipt_ids: List[str],
                              predicted_probabilities: Dict[str, np.array],
                              coicop_mapping: Optional[pd.DataFrame] = None
                              ) -> CoicopOutputFile:

    coicop_classification = []
    for receipt_id, probabilities in zip(receipt_ids, predicted_probabilities):
        coicop_codes = [
            CoicopClassification(
                code=coicop_code,
                description=get_description(coicop_code, coicop_mapping),
                confidence=probability
            )
            for coicop_code, probability in probabilities.items()
        ]
        classification_result = ProductClassificationResult(
            id=receipt_id, coicop_codes=coicop_codes)
        coicop_classification.append(
            classification_result)

    classification_result = ClassificationResult(result=coicop_classification)
    classification_output = CoicopOutputFile(
        coicop_classification_request=receipt_input.coicop_classification_request,
        receipt=receipt_input.receipt,
        coicop_classification_result=classification_result,
        metadata=dict()
    )

    return classification_output
