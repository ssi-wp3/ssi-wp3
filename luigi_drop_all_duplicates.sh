#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.drop_duplicates  DropDuplicatesForAllFiles --local-scheduler

