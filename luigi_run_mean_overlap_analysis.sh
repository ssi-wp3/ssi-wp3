#!/bin/bash
PYTHONPATH="." luigi --module ssi.analysis.tasks OverlapPerPreprocessing --local-scheduler
