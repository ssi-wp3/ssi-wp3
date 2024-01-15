from ssi.coicop_json_parser import *
import unittest


class CoicopJsonParserTest(unittest.TestCase):
    def test_receipt_item_serialized_to_json(self):
        expected_json = {
            "id": "1",
            "description": "Test",
            "quantity": 1,
            "unit_price": 1.0,
            "total_price": 1.0
        }

        receipt_item = ReceiptItem(
            id="1",
            description="Test",
            quantity=1,
            unit_price=1.0,
            total_price=1.0
        )
        self.assertEqual(expected_json, receipt_item.model_dump())

    def test_receipt_item_read_from_json(self):
        json = {
            "id": "1",
            "description": "Test",
            "quantity": 1,
            "unit_price": 1.0,
            "total_price": 1.0
        }
        receipt_item = ReceiptItem.model_validate(json)
        self.assertEqual(json, receipt_item.model_dump())

    def test_receipt_serialized_to_json(self):
        expected_json = {
            "store": "Test",
            "date": "2021-01-01",
            "items": [
                {
                    "id": "1",
                    "description": "item1",
                    "quantity": 1,
                    "unit_price": 1.0,
                    "total_price": 1.0
                },
                {
                    "id": "2",
                    "description": "item2",
                    "quantity": 2,
                    "unit_price": 2.5,
                    "total_price": 5.0
                }
            ]
        }

        receipt = Receipt(
            store="Test",
            date=date(2021, 1, 1),
            items=[
                ReceiptItem(
                    id="1",
                    description="item1",
                    quantity=1,
                    unit_price=1.0,
                    total_price=1.0
                ),
                ReceiptItem(
                    id="2",
                    description="item2",
                    quantity=2,
                    unit_price=2.5,
                    total_price=5.0
                )
            ]
        )
        self.assertEqual(expected_json, receipt.model_dump())

    def test_receipt_read_from_json(self):
        json = {
            "store": "Test",
            "date": "2021-01-01",
            "items": [
                {
                    "id": "1",
                    "description": "item1",
                    "quantity": 1,
                    "unit_price": 1.0,
                    "total_price": 1.0
                },
                {
                    "id": "2",
                    "description": "item2",
                    "quantity": 2,
                    "unit_price": 2.5,
                    "total_price": 5.0
                }
            ]
        }
        receipt = Receipt.model_validate(json)
        self.assertEqual(json, receipt.model_dump())

    def test_coicop_input_file_serialized_to_json(self):
        expected_json = {
            "coicop_classification_request": ["1", "2"],
            "receipt": {
                "store": "supermarket1",
                "date": "2021-01-01",
                "items": [
                    {
                        "id": "1",
                        "description": "item1",
                        "quantity": 1,
                        "unit_price": 1.0,
                        "total_price": 1.0
                    },
                    {
                        "id": "2",
                        "description": "item2",
                        "quantity": 2,
                        "unit_price": 2.5,
                        "total_price": 5.0
                    }
                ]
            },
            "total": 10.0,
            "currency": "EUR",
            "language_hint": "en",
            "metadata": {
                "key": "value"
            }
        }

        coicop_input_file = CoicopInputFile(
            coicop_classification_request=["1", "2"],
            receipt=Receipt(
                store="supermarket1",
                date=date(2021, 1, 1),
                items=[
                    ReceiptItem(
                        id="1",
                        description="item1",
                        quantity=1,
                        unit_price=1.0,
                        total_price=1.0
                    ),
                    ReceiptItem(
                        id="2",
                        description="item2",
                        quantity=2,
                        unit_price=2.5,
                        total_price=5.0
                    )
                ]
            ),
            total=10.0,
            currency="EUR",
            language_hint="en",
            metadata={
                "key": "value"
            }
        )
        self.assertEqual(expected_json, coicop_input_file.model_dump())

    def test_coicop_input_file_read_from_json(self):
        json = {
            "coicop_classification_request": ["1", "2"],
            "receipt": {
                "store": "supermarket1",
                "date": "2021-01-01",
                "items": [
                    {
                        "id": "1",
                        "description": "item1",
                        "quantity": 1,
                        "unit_price": 1.0,
                        "total_price": 1.0
                    },
                    {
                        "id": "2",
                        "description": "item2",
                        "quantity": 2,
                        "unit_price": 2.5,
                        "total_price": 5.0
                    }
                ]
            },
            "total": 10.0,
            "currency": "EUR",
            "language_hint": "en",
            "metadata": {
                "key": "value"
            }
        }
        coicop_input_file = CoicopInputFile.model_validate(json)
        self.assertEqual(json, coicop_input_file.model_dump())

    def test_coicop_classification_serialized_to_json(self):
        expected_json = {
            "code": "1",
            "description": "item1",
            "confidence": 0.5
        }

        coicop_classification = CoicopClassification(
            code="1",
            description="item1",
            confidence=0.5
        )
        self.assertEqual(expected_json, coicop_classification.model_dump())

    def test_coicop_classification_read_from_json(self):
        json = {
            "code": "1",
            "description": "item1",
            "confidence": 0.5
        }
        coicop_classification = CoicopClassification.model_validate(json)
        self.assertEqual(json, coicop_classification.model_dump())

    def test_product_classification_result_serialized_to_json(self):
        expected_json = {
            "id": "1",
            "coicop_codes": [
                {
                    "code": "1",
                    "description": "coicop1",
                    "confidence": 0.75
                },
                {
                    "code": "2",
                    "description": "coicop2",
                    "confidence": 0.85
                }
            ]
        }

        product_classification_result = ProductClassificationResult(
            id="1",
            coicop_codes=[
                CoicopClassification(
                    code="1",
                    description="coicop1",
                    confidence=0.75
                ),
                CoicopClassification(
                    code="2",
                    description="coicop2",
                    confidence=0.85
                )
            ]
        )
        self.assertEqual(
            expected_json, product_classification_result.model_dump())

    def test_product_classification_result_read_from_json(self):
        json = {
            "id": "1",
            "coicop_codes": [
                {
                    "code": "1",
                    "description": "coicop1",
                    "confidence": 0.75
                },
                {
                    "code": "2",
                    "description": "coicop2",
                    "confidence": 0.85
                }
            ]
        }
        product_classification_result = ProductClassificationResult.model_validate(
            json)
        self.assertEqual(json, product_classification_result.model_dump())

    def test_classification_result_serialized_to_json(self):
        expected_json = {
            "result": [
                {
                    "id": "1",
                    "coicop_codes": [
                        {
                            "code": "1",
                            "description": "coicop1",
                            "confidence": 0.75
                        },
                        {
                            "code": "2",
                            "description": "coicop2",
                            "confidence": 0.85
                        }
                    ]
                },
                {
                    "id": "2",
                    "coicop_codes": [
                        {
                            "code": "3",
                            "description": "coicop3",
                            "confidence": 0.65
                        },
                        {
                            "code": "4",
                            "description": "coicop4",
                            "confidence": 0.95
                        }
                    ]
                }
            ]
        }

        classification_result = ClassificationResult(
            result=[
                ProductClassificationResult(
                    id="1",
                    coicop_codes=[
                        CoicopClassification(
                            code="1",
                            description="coicop1",
                            confidence=0.75
                        ),
                        CoicopClassification(
                            code="2",
                            description="coicop2",
                            confidence=0.85
                        )
                    ]
                ),
                ProductClassificationResult(
                    id="2",
                    coicop_codes=[
                        CoicopClassification(
                            code="3",
                            description="coicop3",
                            confidence=0.65
                        ),
                        CoicopClassification(
                            code="4",
                            description="coicop4",
                            confidence=0.95
                        )
                    ]
                )
            ]
        )
        self.assertEqual(expected_json, classification_result.model_dump())
