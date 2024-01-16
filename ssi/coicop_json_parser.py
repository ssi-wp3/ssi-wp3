
from pydantic import BaseModel, field_serializer
from datetime import date
from typing import List, Optional


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

    @field_serializer('date')
    def serialize_date(self, date: date, _info):
        return date.strftime('%Y-%m-%d')


class CoicopInputFile(BaseModel):
    coicop_classification_request: List[str]
    receipt: Receipt
    total: Optional[float]
    currency: Optional[str]
    language_hint: Optional[str]
    metadata: Optional[dict]


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