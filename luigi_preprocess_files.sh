#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.tasks PreprocessCombinedFile --local-scheduler 
