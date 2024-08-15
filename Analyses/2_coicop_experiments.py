import os
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

# local imports
import config
from MLExperiment import MLExperiment
import experiment_parameters as exp_params

class CoicopExperiment:
  def __init__(self, experiment: MLExperiment, stores_in_dev: list[str], stores_in_test: list[str], n_train_samples: int = 5):
    self.experiment = experiment
    self.stores_in_dev = stores_in_dev
    self.stores_in_test = stores_in_test
    self.n_train_samples = n_train_samples

  def eval_pipeline(self, df_dev: pd.DataFrame, df_test: pd.DataFrame):
    df_dev  = df_dev[df_dev["store_name"].isin(self.stores_in_dev)]
    df_test = df_test[df_test["store_name"].isin(self.stores_in_test)]

    X_dev, y_dev = _get_X_y(df_dev, self.experiment.predict_level)
    X_test, y_test = _get_X_y(df_test, self.experiment.predict_level)

    # sample params
    sample_size = 10_000
    sample_replace = True
    
    sample_results: list[dict] = []
    for sample_idx in range(self.n_train_samples):
      sample_seed = config.SEED + sample_idx # have different seed for each sample, otherwise we'll get same samples
      X_dev_, y_dev_ = resample(X_dev, y_dev, replace=sample_replace, random_state=sample_seed)

      self.experiment.eval_pipeline(X_dev_, y_dev_, X_test, y_test, hierarchical_split_func=_get_coicop_level_label)

      sample_result = self.experiment.results
      sample_results.append(sample_result)
    
    # get the mean of all sample results
    sample_results_means: dict = sample_results[0].copy()
    for sample_result in sample_results:
      for key, value in sample_result.items():
        if value is not None:
          sample_results_means[key] += value

    import pdb; pdb.set_trace()
    sample_results_means = {key: value / self.n_train_samples for key, value in sample_results_means.items() if value is not None}
    self.experiment.results = sample_results_means
    import pdb; pdb.set_trace()
    
    stores_in_data = {
      "stores_in_dev" : ', '.join(self.stores_in_dev),
      "stores_in_test": ', '.join(self.stores_in_test)
    }
    self.experiment.metadata.update(stores_in_data)

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

def make_coicop_experiments(pipeline, param_grid: dict, predict_level: int, sample_weight_col_name: str) -> list[CoicopExperiment]:
  ret = []

  param_combinations = ParameterGrid(param_grid)

  for params in param_combinations:
    pipeline_ = clone(pipeline)
    pipeline_.set_params(**params)
    base_experiment = MLExperiment(pipeline_, predict_level, sample_weight_col_name)

    # dev: store 1, test: store 2
    for store_1, store_2 in permutations(config.STORES, 2):
      exp_one_on_one = CoicopExperiment(base_experiment, stores_in_dev=[store_1], stores_in_test=[store_2])
      ret.append(exp_one_on_one)

    # dev: all stores, test: all stores
    exp_all_on_all = CoicopExperiment(base_experiment, stores_in_dev=config.STORES, stores_in_test=config.STORES)
    ret.append(exp_all_on_all)

    # dev: (all stores) - store, test: store 
    for store in config.STORES:
      other_stores = config.STORES.copy()
      other_stores.remove(store)

      exp_rest_on_one = CoicopExperiment(base_experiment, stores_in_dev=other_stores, stores_in_test=[store])
      ret.append(exp_rest_on_one)

  return ret

class testtt:
  def __init__(self):
    pass

if __name__ == "__main__":
  import pdb; pdb.set_trace()
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  results_fn = "results.csv"

  base_pipeline = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True)),
    ("clf", SGDClassifier(n_jobs=8, random_state=config.SEED))]
  )

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  coicop_experiments = make_coicop_experiments(
    exp_params.pipeline,
    exp_params.param_grid,
    exp_params.predict_level,
    exp_params.sample_weight_col_name,
  )

  while len(coicop_experiments) > 0:
    exp = coicop_experiments.pop(0)

    exp.eval_pipeline(df_dev, df_test)
    exp.write_results(out_fn=results_fn)

