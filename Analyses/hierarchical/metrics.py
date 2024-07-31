import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def hierarchical_precision_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable, predict_level: int, average="binary") -> float:
  y_true_all_levels = _concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = _concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return precision_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)

def hierarchical_recall_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable, predict_level: int, average="binary") -> float:
  y_true_all_levels = _concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = _concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return recall_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)

def hierarchical_f1_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable, predict_level: int, average="binary") -> float:
  y_true_all_levels = _concat_upper_levels(y_true, get_upper_level_func, predict_level)
  y_pred_all_levels = _concat_upper_levels(y_pred, get_upper_level_func, predict_level)

  return f1_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)

def _concat_upper_levels(y: np.ndarray, get_upper_level_func: callable, predict_level: int) -> np.ndarray:
  ret = [y]

  for level in range(1, predict_level):
    upper_level_y = get_upper_level_func(y, level)
    ret.append(upper_level_y)

  ret = np.hstack(ret)
  return ret
