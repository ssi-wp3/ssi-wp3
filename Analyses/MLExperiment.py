import os
import csv
import pandas as pd
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score

class MLExperiment:
  def __init__(self, estimator: BaseEstimator, predict_level: int, sample_weight_col_name: str) -> None:
    self.estimator = estimator 
    self.predict_level = predict_level
    self.sample_weight_col_name = sample_weight_col_name

    self.results: dict = {}

  def eval_estimator(self, X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, hierarchical_split_func: callable = None) -> None:
    # todo: add support for sample weight during fit
    self.estimator.fit(X_dev, y_dev)

    y_pred = self.estimator.predict(X_test)

    estimator_name = [str(step) for step in self.estimator.named_steps.values()]
    estimator_name = ', '.join(estimator_name)

    results = {
      "estimator"             : estimator_name,
      "datetime"              : datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
      "predict_level"         : self.predict_level,
      "accuracy"              : accuracy_score(y_test, y_pred),
      "balanced_accuracy"     : balanced_accuracy_score(y_test, y_pred),
      "precision"             : precision_score(y_test, y_pred, average="macro", zero_division=1),
      "recall"                : recall_score(y_test, y_pred, average="macro", zero_division=1),
      "f1"                    : f1_score(y_test, y_pred, average="macro", zero_division=1),
    })

#    if hasattr(self.estimator, "predict_proba"):
#      y_proba = self.estimator.predict_proba(X_test)
#
#      proba_metrics = {
#        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")
#      }
#      self.results.update(proba_metrics)

    if hierarchical_split_func is not None:
      hierarchical_scores = {
        "hierarchical_precision": hierarchical_precision_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro"),
        "hierarchical_recall"   : hierarchical_recall_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro"),
        "hierarchical_f1"       : hierarchical_f1_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro"),
      }

      results.update(hierarchical_scores)
    
    self.results.update(results)

  def write_results(self, out_fn: str) -> None:
    if len(self.results) == 0:
      print("Experiment has not been evaluated! No results will be written.")
      return

    out_exists = os.path.isfile(out_fn)
    with open(out_fn, "a+") as fp:
      writer = csv.DictWriter(fp, delimiter=',', fieldnames=self.results.keys())

      if not out_exists:
        writer.writeheader()

      writer.writerow(self.results)


