from itertools import product

import config
from Experiment import Experiment

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

store_combos = product(config.STORES, repeat=2)
experiments: list[Experiment] = []

# add external validation on stores
for store_1, store_2 in store_combos:
  exp_store_1_store_2 = Experiment(
    pipeline=Pipeline([("hv", HashingVectorizer(input="content", binary=True)), ("clf", SGDClassifier(loss="log_loss", n_jobs=8, random_state=config.SEED))]),
    predict_level=5,
    stores_in_dev=[store_1],
    stores_in_test=[store_2],
    sample_weight=None
  )
  
  experiments.append(exp_store_1_store_2)


## add external validation on stores
#for test_store in config.STORES:
#  dev_stores = config.STORES.copy()
#  dev_stores.remove(test_store)
#
#  exp = Experiment(
#    pipeline=Pipeline([("hv", HashingVectorizer(input="content", binary=True, dtype=bool)), ("clf", SGDClassifier(loss="perceptron", n_jobs=8, random_state=config.SEED))]),
#    predict_level=5,
#    stores_in_dev=dev_stores,
#    stores_in_test=[test_store],
#    sample_weight=None
#  )
#  
#  experiments.append(exp)
