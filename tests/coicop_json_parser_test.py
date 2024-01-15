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

        receipt_item = ReceiptItem(**json)
        self.assertEqual(json, receipt_item.model_dump())
