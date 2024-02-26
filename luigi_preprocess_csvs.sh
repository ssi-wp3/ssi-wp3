#!/bin/bash

PYTHONPATH="." luigi --module ssi.preprocessing.tasks ConvertCPIFiles --input-directory=$1 --output-directory=$2 --local-scheduler 
