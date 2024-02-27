#!/bin/bash
# The CPI files use a weird encoding, convert the files first to utf-8 and convert them to parquet later
# Read all files in the input directory, skip the first 3 bytes (BOM), convert them from cp1252 to utf-8
# and filter out the carriage returns '\r' and write the result to the output directory
input_filename=$1
output_filename=$2

mkdir -p $(dirname $output_filename)
tail -c +4 $input_filename | iconv -f cp1252 -t utf-8 | tr -d '\r' > $output_filename
