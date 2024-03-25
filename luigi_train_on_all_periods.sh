#!/bin/bash
PYTHONPATH="." luigi --module ssi.machine_learning.cross_period TrainModelOnAllPeriods --local-scheduler
