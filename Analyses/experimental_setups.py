from Experiment import Experiment
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

import config

experiments = [
  Experiment(
    pipeline=Pipeline([("hv", HashingVectorizer(input="content", binary=True, dtype=bool)), ("clf", SGDClassifier(loss="perceptron", n_jobs=8, random_state=config.SEED))]),
    predict_level=5,
    stores_in_dev=["ah"],
    stores_in_test=["ah"],
    sample_weight=None
  ),
]

