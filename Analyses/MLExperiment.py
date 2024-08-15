import copy
import os
import csv
import pandas as pd
from datetime import datetime

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score

class MLExperiment:
  def __init__(self, pipeline: Pipeline, predict_level: int, sample_weight_col_name: str, **kwargs) -> None:
    self.pipeline = pipeline 
    self.predict_level = predict_level
    self.sample_weight_col_name = sample_weight_col_name

    pipeline_name = [str(step) for step in self.pipeline.named_steps.values()]
    pipeline_name = ', '.join(pipeline_name)

    self.metadata = {
      "pipeline": pipeline_name,
      "predict_level": predict_level,
      "fit_time": None,
    }
    self.metadata.update(kwargs)

    results_metrics = [
      "accuracy",
      "balanced_accuracy",
      "precision",
      "recall",
      "f1",
      "roc_auc",
      "hierarchical_precision",
      "hierarchical_recall",
      "hierarchical_f1",
    ]

    self._results = dict.fromkeys(results_metrics)

  def eval_pipeline(self, X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, hierarchical_split_func: callable = None) -> None:
    # todo: add support for sample weight during fit
    self.pipeline.fit(X_dev, y_dev)

    y_pred = self.pipeline.predict(X_test)

    self.metadata["fit_time"] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    exp_results = dict()

    exp_results["accuracy"] = accuracy_score(y_test, y_pred)
    exp_results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
    exp_results["precision"] = precision_score(y_test, y_pred, average="macro", zero_division=1)
    exp_results["recall"] = recall_score(y_test, y_pred, average="macro", zero_division=1)
    exp_results["f1"] = f1_score(y_test, y_pred, average="macro", zero_division=1)

    if hierarchical_split_func is not None:
      exp_results["hierarchical_precision"] = hierarchical_precision_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro")
      exp_results["hierarchical_recall"] = hierarchical_recall_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro")
      exp_results["hierarchical_f1"] = hierarchical_f1_score(y_test, y_pred, hierarchical_split_func, self.predict_level, average="macro")

#    if hasattr(self.pipeline, "predict_proba"):
#      y_proba = self.pipeline.predict_proba(X_test)
#      exp_results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    assert all(exp_metric in self._results.keys() for exp_metric in exp_results), "Found key(s) not declared in self._results."
    self._results.update(exp_results)

  def write_results(self, out_fn: str) -> None:
    if len(self._results) == 0:
      print("Experiment has not been evaluated! No results written.")
      return

    out = self.metadata | self._results

    out_exists = os.path.isfile(out_fn)
    with open(out_fn, "a+") as fp:
      writer = csv.DictWriter(fp, delimiter=',', fieldnames=out.keys())

      if not out_exists:
        writer.writeheader()

      writer.writerow(out)
  
  def __copy__(self):
    pipeline_copy = clone(self.pipeline)
    ret = type(self)(pipeline_copy, self.predict_level, self.sample_weight_col_name)
    ret.metadata.update(self.metadata)
    return ret


class BootstrapExperiment:
  def __init__(self, ml_experiment: MLExperiment, n_estimators=5, sample_size=10_000, random_state: int = None):
    self.base_experiment = ml_experiment
    self.n_estimators = n_estimators
    self.sample_size = sample_size
    self.random_state = random_state
    self.experiments: list[MLExperiment] = []
  
  def eval_pipeline(self, X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, hierarchical_split_func: callable = None) -> None:
    for sample_idx in range(self.n_estimators):
      sample_exp = copy.copy(self.base_experiment)
      sample_seed = self.random_state + sample_idx

      X_dev_, y_dev_ = resample(X_dev, y_dev, n_samples=self.n_estimators, random_state=sample_seed)

      sample_exp.eval_pipeline(X_dev_, y_dev_, X_test, y_test, hierarchical_split_func)
      sample_exp.metadata["sample_index"] = sample_idx

      self.experiments.append(sample_exp)

  def write_results(self, out_fn: str) -> None:
    if len(self.experiments) == 0:
      print("No experiments have been evaluated yet.")
      return

    for exp in self.experiments:
      exp.write_results(out_fn)

  def __copy__(self):
    base_exp_copy = copy.copy(self.base_experiment)
    return type(self)(base_exp_copy, self.n_estimators, self.sample_size, self.random_state)
