# %%
import evaluate
from transformers import Trainer
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from functools import partial
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple
import pandas as pd
import argparse
import os
import json

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename",
                    default="/netappdata/ssi_tdjg/data/ssi/feature_extraction/ssi_hf_labse_unique_values.parquet")
parser.add_argument("-o", "--output-directory",
                    type=str, default="./hf_output")
parser.add_argument("-m", "--model-name", type=str,
                    default="sentence-transformers/LaBSE", help="Huggingface sentence transformers model name")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-ic", "--input-column", type=str, default="receipt_text")
parser.add_argument("-lc", "--label-column", type=str,
                    default="coicop_level_1")
parser.add_argument("-ef", "--evaluation-function", type=str, default="f1")
parser.add_argument("-es", "--evaluation-strategy", type=str, default="epoch")
args = parser.parse_args()

hf_labse_features = pd.read_parquet(
    args.input_filename, engine="pyarrow")
hf_labse_features = hf_labse_features[[args.input_column, args.label_column]]
hf_labse_features.head()

# From: https://huggingface.co/docs/transformers/training


def split_data(dataframe: pd.DataFrame, coicop_level: str = "coicop_level_1", val_size: float = 0.1, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    train_val_dataframe, test_dataframe = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe[coicop_level], random_state=random_state)

    train_val_dataframe["label"] = train_val_dataframe[coicop_level]
    train_dataframe, val_dataframe = train_test_split(
        train_val_dataframe, test_size=val_size, stratify=train_val_dataframe[coicop_level], random_state=random_state)

    train_df = Dataset.from_pandas(train_dataframe)
    train_df = train_df.class_encode_column("label")

    val_df = Dataset.from_pandas(val_dataframe)
    val_df = val_df.class_encode_column("label")

    test_dataframe["label"] = test_dataframe[coicop_level]
    test_df = Dataset.from_pandas(test_dataframe)
    test_df = test_df.class_encode_column("label")

    return train_df, val_df, test_df


def tokenize_function(data, text_column: str = "receipt_text", padding: str = "max_length", truncation=True):
    receipt_texts = data[text_column]
    tokens = tokenizer(receipt_texts, padding="max_length")
    return tokens


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


train_df, val_df, test_df = split_data(
    hf_labse_features, coicop_level=args.label_column)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

map_function = partial(tokenize_function, text_column=args.input_column)

train_df = train_df.map(map_function, batched=True)
val_df = val_df.map(map_function, batched=True)
test_df = test_df.map(map_function, batched=True)

train_df = train_df.remove_columns([args.input_column])
val_df = val_df.remove_columns([args.input_column])
test_df = test_df.remove_columns([args.input_column])

print(
    f"Train dataset length: {len(train_df)}, Val dataset length: {len(val_df)}, Test dataset length: {len(test_df)}")
number_of_categories = hf_labse_features[args.label_column].nunique()
number_of_categories

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, num_labels=number_of_categories)

# Create output directory
model_directory = os.path.join(args.output_directory, args.model_name)
date = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
model_directory = os.path.join(model_directory, date)

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

training_args = TrainingArguments(
    output_dir=model_directory,
    evaluation_strategy=args.evaluation_strategy,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size
)

metric = evaluate.load(args.evaluation_function)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=val_df,
    compute_metrics=compute_metrics
)

trainer.train()

final_result_directory = os.path.join(model_directory, "final")
if not os.path.exists(final_result_directory):
    os.makedirs(final_result_directory)

trainer.save_metrics(os.path.join(final_result_directory, "metrics.json"))
trainer.save_model(final_result_directory)
trainer.save_state()

y_pred, labels, metrics = trainer.predict(test_df)
y_true = test_df["label"]

print(classification_report(y_true, y_pred, labels=labels))

with open(os.path.join(final_result_directory, "classification_report.json"), "w") as json_file:
    json_file.write(classification_report(
        y_true, y_pred, labels=labels, output_dict=True))
