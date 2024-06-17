#!/bin/bash
PYTHONPATH="." luigi --module ssi.preprocessing.convert ConvertAllCSVToParquet --local-scheduler 
PYTHONPATH="." luigi --module ssi.preprocessing.convert ConvertAHReceipts --local-scheduler 
PYTHONPATH="." luigi --module ssi.preprocessing.convert ConvertAllJumboReceipts --local-scheduler 
