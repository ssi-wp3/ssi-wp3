import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def hierarchical_precision_score(y_true: pd.Series, y_pred: pd.Series, get_upper_level_func: callable, predict_level: int) -> float:
  y_true_all_levels = __concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = __concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return recall_score(y_true_all_levels, y_pred_all_levels, average="macro")

def hierarchical_precision(y_true: pd.Series, y_pred: pd.Series, get_upper_level_func: callable, predict_level: int) -> float:
  y_true_all_levels = __concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = __concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return f1_score(y_true_all_levels, y_pred_all_levels, average="macro")

def hierarchical_precision_score(y_true: pd.Series, y_pred: pd.Series, get_upper_level_func: callable, predict_level: int) -> float:
  y_true_all_levels = __concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = __concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return precision_score(y_true_all_levels, y_pred_all_levels, average="macro")

def __concat_upper_levels(y: pd.Series, get_upper_level_func: callable, predict_level: int):
  ret = []

  for level in range(1, predict_level):
    upper_level_y = get_upper_level_func(y, level)
    ret.append(upper_level_y)

  ret = pd.concat(ret, axis=0)
  return ret
