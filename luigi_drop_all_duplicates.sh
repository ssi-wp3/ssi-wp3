#!/bin/bash
PYTHONPATH="." luigi --module ssi.feature_extraction.drop_duplicates  DropDuplicatesForAllFiles --local-scheduler

