#!/bin/bash

PYTHONPATH="." luigi --module ssi.preprocessing.clean CleanAllCPIFiles --input-directory=$1 --output-directory=$2 --local-scheduler 
