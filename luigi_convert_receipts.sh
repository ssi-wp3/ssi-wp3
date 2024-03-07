#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.convert ConvertCSVToParquet --input-filename="${data_directory}/preprocessing/cleaned/receipts_plus_202201_Prd.csv" --output-directory="${data_directory}/preprocessing/parquet" --local-scheduler
PYTHONPATH="." luigi --module ssi.preprocessing.convert ConvertAHReceipts --input-filename=$1 --local-scheduler 
