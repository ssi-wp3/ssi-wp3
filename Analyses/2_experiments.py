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

def get_X_y(df: pd.DataFrame, predict_coicop_level: int, coicop_code=None) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_col = f"coicop_level_{predict_coicop_level}"

  if coicop_code is not None and predict_coicop_level > 1:
    parent_coicop_level = predict_coicop_level - 1
    df = df[df[f"coicop_level_{parent_coicop_level}"] == coicop_code]

  X = df[X_col]
  y = df[y_col]

  return X, y

def eval_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, sample_weight=None) -> None:
  #pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  y_pred  = pipe.predict(X_test)
  #y_proba = pipe.predict_proba(X_test)

  print(accuracy_score(y_test, y_pred))
  print(balanced_accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))


def fit_hierarchical(df: pd.DataFrame, predict_coicop_level) -> list[Pipeline]:
  if not ((1 <= predict_coicop_level) and (predict_coicop_level <= 5)):
    print("Predict_coicop_level must be between 1 and 5")
    return

  if predict_coicop_level == 1:
    df_parent_coicop = [("root", df)]
  else:
    parent_coicop_level = predict_coicop_level - 1 
    df_parent_coicop = df.groupby(f"coicop_level_{parent_coicop_level}")

  models = []
  for parent_coicop, df_parent_coicop in df_parent_coicop:

    if df_parent_coicop.empty: continue

    X_dev, y_dev = get_X_y(df_parent_coicop, predict_coicop_level=predict_coicop_level)

    if y_dev.nunique() < 2: continue

    pipe = Pipeline([
      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool, analyzer="word")),
      #("clf", SGDClassifier(n_jobs=6, loss="perceptron", penalty="l2")),
      #("clf", BaggingClassifier(estimator=SGDClassifier, loss="perceptron"), n_jobs=4),
    ])

    pipe.fit(X_dev, y_dev)

    models.append((parent_coicop, pipe))

  return models
    

if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  coicop = 1

#  models = fit_hierarchical(df_dev, coicop)
#
#  for parent_coicop, model in models:
#    X_test, y_test = get_X_y(df_test_1, coicop, parent_coicop)
#    eval_pipeline(model, X_test, y_test)
#    X_test, y_test = get_X_y(df_test_2, coicop, parent_coicop)
#    eval_pipeline(model, X_test, y_test)
#    X_test, y_test = get_X_y(df_test_3, coicop, parent_coicop)
#    eval_pipeline(model, X_test, y_test)
#
  sample_weight = None

  #df_dev = df_dev.sample(frac=1.0, replace=True)

  X_dev, y_dev = get_X_y(df_dev, coicop)
  #X_test, y_test = get_X_y(df_test)

  pipe = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
    #("clf", SGDClassifier(n_jobs=6, loss="perceptron")),
    ("clf", BaggingClassifier(estimator=SGDClassifier(loss="log_loss"), n_jobs=6, n_estimators=100, max_samples=0.3)),
  ])

  X_test, y_test = get_X_y(df_test, coicop)
  pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)
  
  eval_pipeline(pipe, X_test, y_test)


