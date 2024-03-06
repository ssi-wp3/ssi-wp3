#!/bin/bash
PYTHONPATH="." luigi --module ssi.feature_extraction.feature_extraction_tasks class ExtractFeaturesForAllFiles --feature_extraction_method $1 --local-scheduler
