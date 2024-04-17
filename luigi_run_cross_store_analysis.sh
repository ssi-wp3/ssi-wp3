#!/bin/bash
PYTHONPATH="." luigi --module ssi.analysis.tasks CrossStoreAnalysis --local-scheduler
