#!/bin/bash
PYTHONPATH="." luigi --module ssi.machine_learning.adversarial TrainAllAdversarialModels --feature_extractor $1 --local-scheduler
