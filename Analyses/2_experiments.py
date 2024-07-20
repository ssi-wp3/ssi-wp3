import os
from typing import Any, Self
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

import config


def eval_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, sample_weight=None) -> None:
  #pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  y_pred  = pipe.predict(X_test)
  #y_proba = pipe.predict_proba(X_test)

  print(accuracy_score(y_test, y_pred))
  print(balanced_accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))

class HierarchicalClassifier():
  def __init__(self, depth):
    self.depth = depth
    self.root_clf = None
    self.clf_dict = dict()

  def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
    y_root = y["coicop_level_1"]

    root_clf = Pipeline([
      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
      ("clf", SGDClassifier(loss="log_loss", n_jobs=6)),
    ])
    root_clf.fit(X, y_root)

    self.root_clf = root_clf

    for level in range(2, self.depth + 1):
      parent_level = level - 1
      parent_labels = pd.concat([X, y], axis=1).groupby(f"coicop_level_{parent_level}")

      for parent_label, parent_df in parent_labels:
        y_level = y[f"coicop_level_{level}"]

        if y_level.nunique() < 2: continue
        
        clf = Pipeline([
          ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
          ("clf", SGDClassifier(loss="log_loss", n_jobs=6)),
        ])
        clf.fit(X, y_level)

        self.clf_dict[parent_label] = clf

    return self

  def predict(self, X):
    root_y_pred = self.root_clf.predict(X)
    root_y_pred = pd.Series(root_y_pred, name="coicop_label")

    df = pd.concat([X, root_y_pred], axis=1)

    for coicop_level in range(2, self.depth + 1):
      y_pred_by_coicop = df.groupby("coicop_label")

      for coicop_label, df_coicop in y_pred_by_coicop:
        clf = self.clf_dict.get(coicop_label)

        if clf is None: continue

        X_coicop = df_coicop.drop("coicop_label", axis=1)

        df.loc[X_coicop.index, "coicop_label"] = clf.predict(X_coicop)

    return df["coicop_label"]

  def predict_proba(self, X):
    pass

def get_X_y(df: pd.DataFrame, predict_depth: int) -> tuple[pd.DataFrame, pd.Series]:
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

  predict_coicop = 5

  X_dev, y_dev = get_X_y(df_dev, predict_coicop)
  X_test, y_test = get_X_y(df_test, predict_coicop)

  hc = HierarchicalClassifier(depth=predict_coicop)
  hc.fit(X_dev, y_dev)

  y_pred = hc.predict(X_dev)

  eval_pipeline(hc, X_test, y_test[f"coicop_level_{predict_coicop}"])


