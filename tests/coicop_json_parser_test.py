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
