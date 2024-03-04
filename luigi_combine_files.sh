#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.combine CombineRevenueAllFiles --input-directory=$1 -output-directory=$2 --local-scheduler 
