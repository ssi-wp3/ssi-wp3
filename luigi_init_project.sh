#!/bin/bash
PYTHONPATH="." luigi --module ssi.tasks CreateProject --input-directory=$1 --project-directory=$2 --local-scheduler 
