#!/bin/bash
PYTHONPATH="." luigi --module ssi.machine_learning.adversarial TrainAllAdversarialModels --local-scheduler
