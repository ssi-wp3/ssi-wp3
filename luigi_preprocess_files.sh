#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.tasks PreprocessAllFiles --local-scheduler 
