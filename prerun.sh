#!/bin/bash
# This script is run by cnvrg to be able to start before an experiment or notebook
export data_directory=/netappdata/ssi_tdjg/data/ssi

# Configure git
git config pull.rebase false

# Create a luigi configuration file
cp luigi.example.cfg luigi.cfg