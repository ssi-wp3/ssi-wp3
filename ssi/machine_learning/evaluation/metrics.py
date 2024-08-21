from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


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
    # @todo do this in numpy only
    ret = y.str.slice(start=0, stop=label_pos_stop)
    ret = ret.to_numpy()
    return ret


def hierarchical_precision_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable = _get_coicop_level_label, predict_level: int = 5, average="binary") -> float:
    y_true_all_levels = _concat_upper_levels(
        y_true, get_upper_level_func, predict_level)
    y_pred_all_levels = _concat_upper_levels(
        y_pred, get_upper_level_func, predict_level)

    return precision_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)


def hierarchical_recall_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable = _get_coicop_level_label, predict_level: int = 5, average="binary") -> float:
    y_true_all_levels = _concat_upper_levels(
        y_true, get_upper_level_func, predict_level)
    y_pred_all_levels = _concat_upper_levels(
        y_pred, get_upper_level_func, predict_level)

    return recall_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)


def hierarchical_f1_score(y_true: np.ndarray, y_pred: np.ndarray, get_upper_level_func: callable = _get_coicop_level_label, predict_level: int = 5, average="binary") -> float:
    y_true_all_levels = _concat_upper_levels(
        y_true, get_upper_level_func, predict_level)
    y_pred_all_levels = _concat_upper_levels(
        y_pred, get_upper_level_func, predict_level)

    return f1_score(y_true_all_levels, y_pred_all_levels, average=average, zero_division=1)


def _concat_upper_levels(y: np.ndarray, get_upper_level_func: callable, predict_level: int) -> np.ndarray:
    ret = [y]

    for level in range(1, predict_level):
        upper_level_y = get_upper_level_func(y, level)
        ret.append(upper_level_y)

    ret = np.hstack(ret)
    return ret
