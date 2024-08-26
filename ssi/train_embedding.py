# %%
from constants import Constants
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, AutoTokenizer
from functools import partial
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple
from machine_learning.evaluate import plot_confusion_matrix
from preprocessing.combine_unique_values import drop_unknown
from constants import Constants
import pandas as pd
import numpy as np
import evaluate
import argparse
import os
import json

# %%
# "/netappdata/ssi_tdjg/data/ssi/"
data_directory = os.getenv("data_directory", default=".")
print(f"Using data directory: {data_directory}")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename",
                    default=os.path.join(
                        data_directory, "feature_extraction/ssi_hf_labse_unique_values.parquet"))
parser.add_argument("-o", "--output-directory",
                    type=str, default=os.path.join(data_directory, "models"))
parser.add_argument("-m", "--model-name", type=str,
                    default="sentence-transformers/LaBSE", help="Huggingface sentence transformers model name")
parser.add_argument("-s", "--sample-size", type=int, default=None,
                    help="Number of samples to use from the total dataset. These samples are split over train, validation and test datasets.")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-ic", "--input-column", type=str,
                    default=Constants.RECEIPT_TEXT_COLUMN)
parser.add_argument("-lc", "--label-column", type=str,
                    default="coicop_level_1")
parser.add_argument("-ef", "--evaluation-function", type=str, default="f1")
parser.add_argument("-es", "--evaluation-strategy", type=str, default="epoch")
parser.add_argument("-u", "--keep-unknown", action="store_true")
args = parser.parse_args()


def write_model_card_markdown(filename: str,
                              args,
                              dataset: pd.DataFrame,
                              train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame):
    model_card_markdown = f"""
# Model card for {args.model_name}

## Model details    
- Model name: {args.model_name}
- Input column: {args.input_column}
- Label column: {args.label_column}
- Evaluation function: {args.evaluation_function}
- Evaluation strategy: {args.evaluation_strategy}
- Keep unknown: {args.keep_unknown}
- Epochs: {args.epochs}
- Batch size: {args.batch_size}
- Output directory: {args.output_directory}

## Dataset details
- Input filename: {args.input_filename}
- Number of total samples: {len(dataset)}
- Training samples: {len(train_df)} ({len(train_df) / len(dataset)}%)
- Validation samples: {len(val_df)} ({len(val_df) / len(dataset)}%)
- Test samples: {len(test_df)} ({len(test_df) / len(dataset)}%)
- Total number of receipt texts: {dataset[args.input_column].nunique()}
- Number of categories: {dataset[args.label_column].nunique()}
"""
    with open(filename, "w") as file:
        file.write(model_card_markdown)


# From: https://huggingface.co/docs/transformers/training

def split_data(dataframe: pd.DataFrame,
               coicop_level: str,
               val_size: float = 0.1,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[Dataset, Dataset]:
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


def tokenize_function(data, text_column: str = Constants.RECEIPT_TEXT_COLUMN, padding: str = "max_length", truncation=True):
    receipt_texts = data[text_column]
    tokens = tokenizer(receipt_texts, padding="max_length")
    return tokens


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


hf_labse_features = pd.read_parquet(
    args.input_filename, engine="pyarrow")
hf_labse_features = hf_labse_features[[args.input_column, args.label_column]]
if not args.keep_unknown:
    hf_labse_features = drop_unknown(
        hf_labse_features, label_column=args.label_column)

sample_size = args.sample_size
if sample_size is not None:
    hf_labse_features = hf_labse_features.sample(sample_size)

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

# Save COICOP label mappings.
# See: https://discuss.huggingface.co/t/get-label-to-id-id-to-label-mapping/11457
label_features = train_df.features["label"]

# Apparently the _int2str does not return a dict but a list, and needs to be converted before saving.
int2label = {index: value for index,
             value in enumerate(label_features._int2str)}
model_config = AutoConfig.from_pretrained(
    args.model_name,
    label2id=label_features._str2int,
    id2label=int2label,
    num_labels=number_of_categories)
print(label_features._int2str)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, config=model_config)


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

metric = evaluate.combine(["f1", "recall", "precision"])

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

tokenizer.save_pretrained(final_result_directory)
# trainer.save_metrics(os.path.join(final_result_directory, "metrics.json"))
trainer.save_model(final_result_directory)
trainer.save_state()

y_pred, y_true_labels, metrics = trainer.predict(test_df)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.array(test_df["label"])
labels = train_df.features["label"].names


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
print(confusion_matrix_df)

print(classification_report(y_true, y_pred, target_names=labels))

confusion_matrix_df.to_csv(os.path.join(
    final_result_directory, "confusion_matrix.csv"), sep=";")

plot_confusion_matrix(y_true, y_pred, os.path.join(
    final_result_directory), labels=labels)

with open(os.path.join(final_result_directory, "classification_report.json"), "w") as json_file:
    json.dump(classification_report(
        y_true, y_pred, target_names=labels, output_dict=True), json_file)

write_model_card_markdown(os.path.join(final_result_directory, "model_card.md"),
                          args, hf_labse_features, train_df, val_df, test_df)
