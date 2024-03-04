#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.preprocess PreprocessAllFiles --local-scheduler 
