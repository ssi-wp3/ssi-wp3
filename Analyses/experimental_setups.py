from itertools import product

import config
from Experiment import Experiment

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

experiments: list[Experiment] = []

#dev_test_combos = [([store_1], [store_2]) for store_1, store_2 in product(config.STORES, repeat=2)] 
#dev_test_combos.append((config.STORES, config.STORES))  # train = test
#dev_test_combos.extend([(list(set(config.STORES) - set([store])), [store]) for store in config.STORES]) # train on n - 1, test on 1
dev_test_combos = []
dev_test_combos.extend([(config.STORES, [store]) for store in config.STORES]) # train on n, test on 1

#analyzers = ["char_wb", "word"]
analyzers = ["word"]
alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
#alphas = [0.0001]
#ngram_ranges = [(3, 3), (1, 1)]
ngram_ranges = [(1, 1)]
n_features = [2**20]

for (stores_in_dev, stores_in_test) in dev_test_combos:
  for analyzer in analyzers:
    for n_feature in n_features:
      for ngram_range in ngram_ranges:
        for alpha in alphas:
          base_pipeline = Pipeline([
          ("hv", HashingVectorizer(input="content", binary=True, ngram_range=ngram_range, dtype=bool, analyzer=analyzer, n_features=n_feature)),
          ("clf", SGDClassifier(loss="log_loss", n_jobs=8, random_state=config.SEED, alpha=alpha))])

          exp_store_1_store_2 = Experiment(
            pipeline=base_pipeline,
            predict_level=5,
            stores_in_dev=stores_in_dev,
            stores_in_test=stores_in_test,
            sample_weight=None
          )

          experiments.append(exp_store_1_store_2)


