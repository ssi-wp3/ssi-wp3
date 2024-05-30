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
from typing import Tuple
import pandas as pd
import argparse
import os

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename",
                    default="/netappdata/ssi_tdjg/data/ssi/feature_extraction/ssi_hf_labse_unique_values.parquet")
parser.add_argument("-o", "--output-directory",
                    type=str, default="./hf_output")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("-ic", "--input-column", type=str, default="receipt_text")
parser.add_argument("-lc", "--label-column", type=str,
                    default="coicop_level_1")
parser.add_argument("-ef", "--evaluation-function", type=str, default="f1")
parser.add_argument("-es", "--evaluation-strategy", type=str, default="epoch")
args = parser.parse_args()


# %%
hf_labse_features_filename = args.input_filename

# %%
hf_labse_features = pd.read_parquet(
    hf_labse_features_filename, engine="pyarrow")
hf_labse_features = hf_labse_features[[args.input_column, args.label_column]]
hf_labse_features.head()

# %%
# From: https://huggingface.co/docs/transformers/training


def split_data(dataframe: pd.DataFrame, coicop_level: str = "coicop_level_1", test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    train_dataframe, test_dataframe = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe[coicop_level], random_state=random_state)

    train_dataframe["label"] = train_dataframe[coicop_level]
    train_df = Dataset.from_pandas(train_dataframe)
    train_df = train_df.class_encode_column("label")

    test_dataframe["label"] = test_dataframe[coicop_level]
    test_df = Dataset.from_pandas(test_dataframe)
    test_df = test_df.class_encode_column("label")

    return train_df, test_df


# %%
train_df, test_df = split_data(
    hf_labse_features, coicop_level=args.label_column)

# %%

model_name = "sentence-transformers/LaBSE"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(data, text_column: str = "receipt_text", padding: str = "max_length", truncation=True):
    receipt_texts = data[text_column]
    tokens = tokenizer(receipt_texts, padding="max_length")
    return tokens


map_function = partial(tokenize_function, text_column=args.input_column)

train_df = train_df.map(map_function, batched=True)
test_df = test_df.map(map_function, batched=True)

# %%
train_df = train_df.remove_columns([args.input_column])
test_df = test_df.remove_columns([args.input_column])

# %%
train_df

# %%
test_df

# %%
number_of_categories = hf_labse_features[args.label_column].nunique()
number_of_categories

# %%

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=number_of_categories)

# %%


training_args = TrainingArguments(
    output_dir=args.output_directory,
    evaluation_strategy=args.evaluation_strategy,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size
)

# %%


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

# %%


metric = evaluate.load(args.evaluation_function, )

# %%

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=test_df,
    compute_metrics=compute_metrics
)

# %%
trainer.train()
