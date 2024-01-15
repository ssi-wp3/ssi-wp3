
from pydantic import BaseModel
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


class CoicopClassification(BaseModel):
    code: str
    description: Optional[str]
    confidence: Optional[float]


class ProductClassificationResult(BaseModel):
    id: str
    coicop_codes: List[CoicopClassification]


class ClassificationResult(BaseModel):
    result: List[ProductClassificationResult]


class CoicopInputFile(BaseModel):
    coicop_classification_request: List[str]
    receipt: Receipt
    total: Optional[float]
    currency: Optional[str]
    language_hint: Optional[str]
    metadata: Optional[dict]


class CoicopOutputFile(CoicopInputFile):
    coicop_classification_result: ClassificationResult
    metadata: Optional[dict]
