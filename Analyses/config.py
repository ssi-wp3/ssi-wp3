import os

SUPERMARKETS = ["lidl", "ah", "jumbo", "plus"]

SOURCE_DATA_DIR = os.path.join('/', "netappdata", "ssi_tdjg", "data", "ssi", "preprocessing", "05-final")
SOURCE_DATA_PATHS = [os.path.join(SOURCE_DATA_DIR, f"ssi_{sm}_revenue.parquet") for sm in SUPERMARKETS]

