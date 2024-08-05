import os

STORES = ["lidl", "ah", "jumbo", "plus"]

SOURCE_DATA_DIR = os.path.join('/', "netappdata", "ssi_tdjg", "data", "ssi", "preprocessing", "05-final")
OUTPUT_DATA_DIR = os.path.join('.', "data")
OUTPUT_GRAPHICS_DIR = os.path.join('.', "graphics")

#SAMPLE_N: int = 10_000 # set None if no sample
SAMPLE_N: int = None # set None if no sample

SEED = 42

