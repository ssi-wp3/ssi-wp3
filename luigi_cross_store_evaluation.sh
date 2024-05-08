#!/bin/bash
PYTHONPATH="." luigi --module ssi.machine_learning.cross_store AllCrossStoreEvaluations --local-scheduler
