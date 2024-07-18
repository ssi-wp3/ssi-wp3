import os
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import config

def get_X_y(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_col = "coicop_level_1"

  X = df[X_col]
  y = df[y_col]

  return X, y

def eval_pipeline(pipe: Pipeline, X_dev, y_dev, X_test, y_test):
  pipe.fit(X_dev, y_dev)

  y_pred  = pipe.predict(X_test)
  y_proba = pipe.predict_proba(X_test)

  print(accuracy_score(y_test, y_pred))
  print(balanced_accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))


if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  if type(sample) == float:
    df_dev = df_dev.sample(frac=0.1, replace=True)

  X_dev, y_dev = get_X_y(df_dev)
  X_test, y_test = get_X_y(df_test)

  pipe = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
    #("clf", DummyClassifier()),
    ("clf", SGDClassifier(n_jobs=4, loss="modified_huber")),
  ])
  
  eval_pipeline(pipe, X_dev, y_dev, X_test, y_test)

