import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score 
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

import config
from hierarchical.models import HierarchicalClassifier 
from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score


def eval_pipeline(pipe: Pipeline, X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, predict_level: int, get_upper_level_func: callable, sample_weight=None) -> None:
  pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  import pdb; pdb.set_trace()

  y_pred = pipe.predict(X_test)
  # y_proba = None

  #
  # write results
  #
  out_fn = "results.csv"

  pipeline_name = [str(step) for step in pipe.named_steps.values()]
  pipeline_name = ', '.join(pipeline_name)

  out = {
    "pipeline"              : pipeline_name,
    "datetime"              : datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
    "predict_level"         : predict_level,
    "accuracy"              : accuracy_score(y_test, y_pred),
    "balanced_accuracy"     : balanced_accuracy_score(y_test, y_pred),
    "precision"             : precision_score(y_test, y_pred, average="macro"),
    "recall"                : recall_score(y_test, y_pred, average="macro"),
    "f1"                    : f1_score(y_test, y_pred, average="macro"),
    "hierarchical_precision": hierarchical_precision_score(y_test, y_pred, get_upper_level_func, predict_level, average="macro"),
    "hierarchical_recall"   : hierarchical_recall_score(y_test, y_pred, get_upper_level_func, predict_level, average="macro"),
    "hierarchical_f1"       : hierarchical_f1_score(y_test, y_pred, get_upper_level_func, predict_level, average="macro"),
  }

  out_exists = os.path.isfile(out_fn)
  with open(out_fn, "a+") as fp:
    writer = csv.DictWriter(fp, delimiter=',', fieldnames=out.keys())

    if not out_exists:
      writer.writeheader()

    writer.writerow(out)
    

def get_X_y(df: pd.DataFrame, predict_level: int) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_col = f"coicop_level_{predict_level}"

  X = df[X_col]
  y = df[y_col]

  return X, y

def get_coicop_level_label(y: pd.Series, level: int) -> pd.Series:
  if not isinstance(y, pd.Series):
    if isinstance(y, np.ndarray):
      y = pd.Series(y)
    elif isinstance(y, pd.DataFrame):
      print("Pandas DataFrame is given instead of Pandas Series.")
      exit(1)
    else:
      print("Unknown type given.")
      exit(1)
     
  if level > 5:
    print("Undefined COICOP level!")
    return

  label_pos_stop = level + 1
  return y.str.slice(start=0, stop=label_pos_stop)

if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  predict_coicop = 5

  X_dev, y_dev = get_X_y(df_dev, predict_coicop)
  X_test, y_test = get_X_y(df_test, predict_coicop)

#  pipe = Pipeline([
#    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#    #("tfidf", TfidfTransformer()),
#    #("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#    ("clf", SGDClassifier(loss="log_loss", n_jobs=6, alpha=0.00000001)),
#  ])
#
#  hc = HierarchicalClassifier(pipe, depth=predict_coicop)
#  hc.fit(X_dev, y_dev, get_coicop_level_label)

  hc = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=False, norm=None, analyzer="char", ngram_range=(3, 6), n_features=150_000)),
    #("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
    ("clf", SGDClassifier(loss="log_loss", n_jobs=6, alpha=0.00000001)),
  ])


  eval_pipeline(hc, X_dev, y_dev, X_test, y_test, predict_level=predict_coicop, get_upper_level_func=get_coicop_level_label)

#  hc = Pipeline([
#      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#      ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#  ])
#  hc.fit(X_dev, y_dev[f"coicop_level_{predict_coicop}"])


