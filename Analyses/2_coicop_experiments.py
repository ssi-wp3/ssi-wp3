import os
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from Experiment import Experiment

import config

class CoicopExperiment:
  def __init__(self, pipeline: Pipeline, predict_level: int, sample_weight: str, stores_in_dev: list[str], stores_in_test: list[str]):
    self.experiment = Experiment(pipeline, predict_level, sample_weight)
    self.stores_in_dev = stores_in_dev
    self.stores_in_test = stores_in_test

  def eval_pipeline(self, df_dev: pd.DataFrame, df_test: pd.DataFrame):
    df_dev  = df_dev["store_name"].isin(self.stores_in_dev)
    df_test = df_test["store_name"].isin(self.stores_in_test)

    X_dev, y_dev = _get_X_y(df_dev, exp.predict_level)
    X_test, y_test = _get_X_y(df_test, exp.predict_level)
    
    self.experiment.eval_pipeline(X_dev, y_dev, X_test, y_test, hierarchical_split_func=get_coicop_level_label)
    
    stores_in_data = {
      "stores_in_dev" : ', '.join(self.stores_in_dev),
      "stores_in_test": ', '.join(self.stores_in_test)
    }
    self.experiment.results.update(stores_in_data)

  def write_results(self, out_fn: str) -> None:
    self.experiment.write_results(out_fn)


def _get_X_y(df: pd.DataFrame, predict_level: int) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_col = f"coicop_level_{predict_level}"

  X = df[X_col]
  y = df[y_col]

  return X, y

def _get_coicop_level_label(y: pd.Series, level: int) -> np.ndarray:
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
  ret = y.str.slice(start=0, stop=label_pos_stop) # @todo do this in numpy only
  ret = ret.to_numpy()
  return ret

if __name__ == "__main__":
  if config.SAMPLE_N is not None:
    df_dev_fn  = "sample_dev_lidl_ah_jumbo_plus.parquet"
    df_test_fn = "sample_test_lidl_ah_jumbo_plus.parquet"

    results_fn = "sample_results.csv"

  else:
    df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
    df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

    results_fn = "results.csv"

  base_pipeline = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True)),
    ("clf", SGDClassifier(n_jobs=8, random_state=config.SEED))]
  )

  import hyperparameters

  for (_, step) in base_pipeline.steps:
    step_class = step.__class__

    step_hp = hyperparameters.clf_hyperparameters[step_class]

  import pdb; pdb.set_trace()

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  while len(experiments) > 0:
    exp = experiments.pop(0)

    exp.eval_pipeline(df_dev, df_test)
    exp.write_results(out_fn=results_fn)

  exit(0)

