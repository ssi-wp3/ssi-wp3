from typing import Dict, Any
import scipy.stats as stats

# Feature extraction parameters

COUNT_VECTORIZER = {
    'lowercase': [True, False],
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'analyzer': ['word', 'char', 'char_wb'],
    # 'min_df': stats.uniform(0.01, 0.1),
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
    'C': stats.loguniform(1e-5, 100),
    'fit_intercept': [True, False],
    'max_iter': [100, 200, 300, 400, 500],
    'solver': ['liblinear', 'saga'],
}


DISTRIBUTIONS_FOR_ALL_MODELS = {
    "LogisticRegression": LOGISTIC_REGRESSION,
}


class ParamDistributions:

    @staticmethod
    def add_prefix_to_dict_keys(dictionary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in dictionary.items()}

    @ staticmethod
    def distributions_for_model(model_type: str, prefix: str = "") -> Dict[str, Any]:
        return ParamDistributions.add_prefix_to_dict_keys(DISTRIBUTIONS_FOR_ALL_MODELS[model_type], prefix)

    @staticmethod
    def distributions_for_feature_extraction(feature_extraction_type: str, prefix: str = "") -> Dict[str, Any]:
        return ParamDistributions.add_prefix_to_dict_keys(FEATURE_EXTRACTION_PARAMS[feature_extraction_type], prefix)

    @staticmethod
    def distributions_for_pipeline(feature_extraction_type: str,
                                   model_type: str,
                                   feature_pipeline_prefix: str = "vectorizer__",
                                   model_prefix: str = "clf__"
                                   ) -> Dict[str, Any]:
        return {**ParamDistributions.distributions_for_feature_extraction(feature_extraction_type, feature_pipeline_prefix),
                **ParamDistributions.distributions_for_model(model_type, model_prefix)}
