import os
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

import config
from custom_classifier import HierarchicalClassifier 


def eval_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, sample_weight=None) -> None:
  #pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  y_pred = pipe.predict(X_test)
  #y_proba = pipe.predict_proba(X_test)

  print(accuracy_score(y_test, y_pred))
  print(balanced_accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))


def get_X_y(df: pd.DataFrame, predict_depth: int) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_cols = [f"coicop_level_{predict_level}" for predict_level in range(1, predict_depth+1)]

  X = df[X_col]
  y = df[y_cols]

  return X, y

if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  predict_coicop = 2

  X_dev, y_dev = get_X_y(df_dev, predict_coicop)
  X_test, y_test = get_X_y(df_test, predict_coicop)

  pipe = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
    ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
  ])

  hc = HierarchicalClassifier(pipe, depth=predict_coicop)
  hc.fit(X_dev, y_dev)

#  hc = Pipeline([
#      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#      ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#  ])
#  hc.fit(X_dev, y_dev[f"coicop_level_{predict_coicop}"])

  eval_pipeline(hc, X_test, y_test[f"coicop_level_{predict_coicop}"])


