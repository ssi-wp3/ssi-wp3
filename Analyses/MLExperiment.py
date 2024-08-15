import os
import csv
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score

class MLExperiment:
  def __init__(self, pipeline: Pipeline, predict_level: int, sample_weight_col_name: str) -> None:
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

    self._results: list[dict] = []

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

    assert all(exp_metric in results_metrics for exp_metric in exp_results), "Found key(s) not declared in self._results."
    self._results.append(exp_results)

  def write_results(self, out_fn: str) -> None:
    if len(self._results) == 0:
      print("Experiment has not been evaluated! No results written.")
      return

    results_df  = pd.DataFrame(self._results, columns=results_metrics)
    metadata_df = pd.DataFrame.from_records([self.metadata], index=results_df.index)
    metadata_df = metadata_df.ffill()

    out = pd.concat([results_df, metadata_df], axis=1)

    out.to_csv(out_fn, mode='a')
#    out_exists = os.path.isfile(out_fn)
#    with open(out_fn, "a+") as fp:
#      writer = csv.DictWriter(fp, delimiter=',', fieldnames=out.keys())
#
#      if not out_exists:
#        writer.writeheader()
#
#      writer.writerow(out)


class BootstrapExperiment:
  def __init__(self, ml_experiment: MLExperiment):
    pass
