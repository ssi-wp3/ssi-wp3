#!/bin/bash
PYTHONPATH="." luigi --module ssi.feature_extraction.feature_extraction_tasks ExtractFeaturesForAllFiles --feature-extraction-method $1 --local-scheduler
