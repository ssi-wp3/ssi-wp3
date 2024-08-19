from typing import Dict, Any
import scipy.stats as stats

# Feature extraction parameters

COUNT_VECTORIZER = {
    'lowercase': [True, False],
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'analyzer': ['word', 'char', 'char_wb'],
    'min_df': stats.uniform(0, 0.2),
    'max_df': stats.uniform(0.8, 0.2),
    'max_features': range(100, 10000),
}

FEATURE_EXTRACTION_PARAMS = {
    'CountVectorizer': COUNT_VECTORIZER,
    'TfidfVectorizer': COUNT_VECTORIZER,
}

# Model parameters

LOGISTIC_REGRESSION = {
    'penalty': ['l1', 'l2'],
    'C': stats.loguniform(0.1, 1),
    'fit_intercept': [True, False],
    'max_iter': [100, 200, 300],
    'solver': ['liblinear', 'saga'],
}


DISTRIBUTIONS_FOR_ALL_MODELS = {
    "LogisticRegression": LOGISTIC_REGRESSION,
}


class ParamDistributions:

    @ staticmethod
    def distributions_for_model(model_type: str) -> Dict[str, Any]:
        return DISTRIBUTIONS_FOR_ALL_MODELS[model_type]

    @staticmethod
    def distributions_for_feature_extraction(feature_extraction_type: str) -> Dict[str, Any]:
        return FEATURE_EXTRACTION_PARAMS[feature_extraction_type]

    @staticmethod
    def distributions_for_pipeline(feature_extraction_type: str,
                                   model_type: str,
                                   feature_pipeline_prefix: str = "vectorizer__",
                                   model_prefix: str = "clf__"
                                   ) -> Dict[str, Any]:
        return {
            **{f"{feature_pipeline_prefix}_{param_name}": param_settings for param_name, param_settings in FEATURE_EXTRACTION_PARAMS[feature_extraction_type].items()},
            **{f"{model_prefix}_{param_name}": param_settings for param_name, param_settings in DISTRIBUTIONS_FOR_ALL_MODELS[model_type].items()}
        }
