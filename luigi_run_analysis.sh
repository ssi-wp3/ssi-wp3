#!/bin/bash
PYTHONPATH="." luigi --module ssi.analysis.tasks AllStoresAnalysis --local-scheduler
