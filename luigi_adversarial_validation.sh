#!/bin/bash
PYTHONPATH="." luigi --module ssi.machine_learning.machine_learning_tasks TrainAllAdversarialModels --local-scheduler
