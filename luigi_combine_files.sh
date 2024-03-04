#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.combine CombineAllRevenueFiles --local-scheduler
