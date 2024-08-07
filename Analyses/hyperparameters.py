import config

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

clf_hyperparameters = {
  HashingVectorizer: {
    "input": ["content"],
    "binary": [True],
    "analyzer": ["word", "char", "char_wb"],
    "n_gram_range": [(1, 1), (3, 3)],
    "n_features": [2**20],
  },
  SGDClassifier: {
    "loss": ["log_loss", "perception"],
    "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
    "sample_weight": [None],
  }
}

