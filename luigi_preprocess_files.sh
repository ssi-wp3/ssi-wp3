#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.preprocess PreprocessAllFiles --input-directory=$1 --output-directory=$2 --local-scheduler 
