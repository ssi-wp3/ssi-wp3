#!/bin/sh

# Trains a model to learn embeddings from a dataset for different COICOP levels
while getopts ":o:m:e:b:" opt; do
  case $opt in
    o) output_dir="$OPTARG"
    ;;
    m) model="$OPTARG"
    ;;
    e) epochs="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done 

# Provide default values
output_dir=${output_dir:-"output"}
model=${model:-"sentence-transformers/paraphrase-MiniLM-L6-v2"}
epochs=${epochs:-6}
batch_size=${batch_size:-96}

for label_column in coicop_level_1 coicop_level_2 coicop_level_3 coicop_level_4 coicop_number
do
    python ssi/train_embedding.py -o $output_dir -m $model -e $epochs -b $batch_size -lc $label_column
done