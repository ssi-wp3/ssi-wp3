#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.receipts AddAllReceiptTexts --local-scheduler
