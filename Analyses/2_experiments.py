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


def eval_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, sample_weight=None) -> None:
  #pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  y_pred  = pipe.predict(X_test)
  #y_proba = pipe.predict_proba(X_test)

  print(accuracy_score(y_test, y_pred))
  print(balanced_accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))

class HierarchicalClassifier():
  def __init__(self, depth=4):
    self.depth = depth
    self.root_clf = None
    self.clf_dict = dict()

  def fit(self, df):
    X_root, y_root = self.get_X_y(df, 1)

    self.root_clf = Pipeline([
      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
      ("clf", SGDClassifier(loss="log_loss", n_jobs=6)),
    ]).fit(X_root, y_root)

    for coicop_level in range(2, self.depth + 1):
      parent_coicop_level = coicop_level - 1
      parent_coicops = df.groupby(f"coicop_level_{parent_coicop_level}")

      for parent_coicop_label, parent_coicop_df in parent_coicops:
        X, y = self.get_X_y(parent_coicop_df, coicop_level)

        if y.nunique() < 2: continue
        
        self.clf_dict[parent_coicop_label] = Pipeline([
          ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
          ("clf", SGDClassifier(loss="log_loss", n_jobs=6)),
        ]).fit(X, y)

  def predict(self, X):
    y_pred = self.root_clf.predict(X)
    y_pred = pd.Series(y_pred, name="coicop_label")

    return self.__predict(X, y_pred)

  def __predict(self, X, parent_y_pred):
    df = pd.concat([X, parent_y_pred], axis=1)

    for coicop_level in range(2, self.depth + 1):
      y_pred_by_coicop = df.groupby("coicop_label")
      for coicop_label, df_coicop in y_pred_by_coicop:
        clf = self.clf_dict.get(coicop_label)

        if clf is None: continue

        X_coicop = df_coicop.drop("coicop_label", axis=1)

        df.loc[X_coicop.index, "coicop_label"] = clf.predict(X_coicop)

    return df

#  def __predict(self, X, parent_coicop_label: str):
#    clf = self.clf_dict.get(parent_coicop_label)
#
#    if clf is None:
#      return parent_coicop_label
#
#    y_pred = clf.predict(X)
#    return self.__predict(X, y_pred)

      

  def predict_proba(self, X):
    pass

  def get_X_y(self, df: pd.DataFrame, predict_coicop_level: int) -> tuple[pd.Series, pd.Series]:
    X_col = "receipt_text" # text column
    y_col = f"coicop_level_{predict_coicop_level}"

    X = df[X_col]
    y = df[y_col]

    return X, y



if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  hc = HierarchicalClassifier()
  hc.fit(df_dev)

  coicop = 4

  X_dev, y_dev   = hc.get_X_y(df_dev, coicop)
  X_test, y_test = hc.get_X_y(df_test, coicop)

  y_pred = hc.predict(X_dev)

  import pdb; pdb.set_trace()



  
  eval_pipeline(pipe, X_test, y_test)


