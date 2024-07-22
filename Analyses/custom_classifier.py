from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline
from typing_extensions import Self
import pandas as pd

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, clf, depth: int) -> None:
    # TODO add multi core (n_jobs)
    # TODO add checks whether clf has functions fit and predict
    # TODO integrate sklearn validation steps 
    # TODO remove the need to identify levels by column name: add function pointer as arg in fit to split the subclasses. This way, fit will still only accept the predict labels
    # TODO propagate through the classification tree by predict_proba
    self.depth = depth
    self.root_clf = clf
    self.clf_dict = dict()

  def fit(self, X: pd.Series, y: pd.DataFrame) -> Self:
    y_root = y["coicop_level_1"]

    self.root_clf.fit(X, y_root)

    for level in range(2, self.depth + 1):
      parent_level = level - 1
      y_by_parent_label = y.groupby(f"coicop_level_{parent_level}")

      for parent_label, predict_y in y_by_parent_label:
        y_predict = predict_y[f"coicop_level_{level}"]
        X_predict = X.loc[predict_y.index]

        if y_predict.nunique() < 2: continue
        
        clf = clone(self.root_clf) # copy clf / pipeline without copying the weights
        clf.fit(X_predict, y_predict)

        self.clf_dict[parent_label] = clf

    return self

  def predict(self, X: pd.Series) -> pd.Series:
    ret = self.root_clf.predict(X)
    ret = pd.Series(ret, name="coicop_label", index=X.index)

    for coicop_level in range(2, self.depth + 1):
      y_pred_by_coicop = ret.groupby(ret) # group by values

      for coicop_label, df_coicop in y_pred_by_coicop:
        clf = self.clf_dict.get(coicop_label)

        if clf is None: continue

        X_coicop = X[df_coicop.index]

        y_pred_coicop = clf.predict(X_coicop)
        ret.loc[X_coicop.index] = y_pred_coicop

    return ret

  def predict_proba(self, X):
    pass
