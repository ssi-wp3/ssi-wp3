import config

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier

predict_level: int = 5
sample_weight_col_name: str = None # not supported yet!

pipeline = Pipeline([
  ("hv", HashingVectorizer(input="content", binary=True)),
  ("clf", SGDClassifier(n_jobs=8, random_state=config.SEED))]
)

param_grid = {
  "hv__input": ["content"],
  "hv__binary": [True],
  "hv__analyzer": ["word", "char", "char_wb"],
  "hv__ngram_range": [(1, 1), (3, 3)],
  "hv__n_features": [2**20],
  "clf__loss": ["log_loss"],
  "clf__alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
}


