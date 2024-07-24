from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from typing_extensions import Self
import pandas as pd

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, clf, depth: int) -> None:
    # TODO add multi core (n_jobs)
    # TODO add checks whether clf has functions fit and predict
    # TODO integrate sklearn validation steps 
    # TODO propagate through the classification tree by predict_proba
    self.depth = depth
    self.root_clf = clf
    self.clf_dict = dict()

  def fit(self, X: pd.Series, y: pd.Series, get_upper_level_func: callable) -> Self:
    y_root = get_upper_level_func(y, 1)

    self.root_clf.fit(X, y_root)

    for level in range(2, self.depth + 1):
      parent_level = level - 1
      y_parent_level = get_upper_level_func(y, parent_level)

      for parent_label, y_parent in y_parent_level.groupby(y_parent_level):
        X_predict = X.loc[y_parent.index]

        y_predict = get_upper_level_func(y, level)
        y_predict = y_predict.loc[y_parent.index]

        if y_predict.nunique() == 0: 
          print("Internal node with no classes!")
          continue
        elif y_predict.nunique() == 1: 
          clf = DummyClassifier(strategy="constant", constant=y_predict.iloc[0])
        else:
          clf = clone(self.root_clf) # copy clf / pipeline without copying the weights

        clf.fit(X_predict, y_predict)

        self.clf_dict[parent_label] = clf

    return self

  def predict(self, X: pd.Series) -> pd.Series:
    ret = self.root_clf.predict(X)
    ret = pd.Series(ret, index=X.index)

    for _ in range(2, self.depth + 1):
      # @todo fix this for loop
      y_pred_by_label = ret.groupby(ret) # group by values

      for label_name, df_label in y_pred_by_label:
        clf = self.clf_dict.get(label_name)

        if clf is None: continue

        X_label = X[df_label.index]

        y_pred_label = clf.predict(X_label)
        ret.loc[X_label.index] = y_pred_label

    return ret

  def predict_proba(self, X: pd.DataFrame):
    pass

