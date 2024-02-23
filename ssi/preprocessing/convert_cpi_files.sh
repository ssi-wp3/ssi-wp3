#!/bin/bash
# The CPI files use a weird encoding, convert the files first to utf-8 and convert them to parquet later
# Read all files in the input directory, skip the first 3 bytes (BOM), convert them from cp1252 to utf-8
# and filter out the carriage returns '\r' and write the result to the output directory
input_directory=$1
output_directory=$2

for file in $input_directory/*.csv 
do
    tail -c +4 $file | iconv -f cp1252 -t utf-8 | tr -d '\r' > $output_directory/$(basename $file)
done
