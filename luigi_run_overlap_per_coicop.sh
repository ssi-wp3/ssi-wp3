#!/bin/bash
PYTHONPATH="." luigi --module ssi.analysis.tasks OverlapPerPreprocessingAndCoicop --local-scheduler
