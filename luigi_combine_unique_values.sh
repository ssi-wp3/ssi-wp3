#!/bin/bash
PYTHONPATH="." luigi --module ssi.feature_extraction.feature_extraction_tasks CombineUniqueValues --local-scheduler
