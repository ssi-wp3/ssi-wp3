from typing import Dict, Any
import scipy.stats as stats
from enum import Enum


class FeatureExtractorType(Enum):
    """ The type of feature extractor.
    """
    count_vectorizer = "CountVectorizer"
    hashing_vectorizer = "HashingVectorizer"
    tfidf_vectorizer = "TfidfVectorizer"


class ModelType(Enum):
    """ The type of model.
    """
    logistic_regression = "LogisticRegression"


# Feature extraction parameters
COUNT_VECTORIZER = {
    'lowercase': [True, False],
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'analyzer': ['word', 'char', 'char_wb'],
    # 'min_df': stats.uniform(0.01, 0.1),
    'max_df': stats.uniform(0.8, 0.2),
    'max_features': range(100, 10000),
}

HASHING_VECTORIZER = {
    "input": ["content"],
    "binary": [True],
    'analyzer': ['word', 'char', 'char_wb'],
    'lowercase': [True, False],
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'norm': ['l1', 'l2'],
}

FEATURE_EXTRACTION_PARAMS = {
    FeatureExtractorType.count_vectorizer.value: COUNT_VECTORIZER,
    FeatureExtractorType.hashing_vectorizer.value: HASHING_VECTORIZER,
    FeatureExtractorType.tfidf_vectorizer.value: COUNT_VECTORIZER,
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
    ModelType.logistic_regression.value: LOGISTIC_REGRESSION,
}


class ParamDistributions:
    """ The class for the parameter distributions.
    """

    @staticmethod
    def add_prefix_to_dict_keys(dictionary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """ Add a prefix to the keys of a dictionary.

        Parameters
        ----------
        dictionary : Dict[str, Any]
            The dictionary to add the prefix to.
        prefix : str
            The prefix to add to the keys.

        Returns
        -------
        Dict[str, Any]
            The dictionary with the prefix added to the keys.
        """
        return {f"{prefix}{key}": value for key, value in dictionary.items()}

    @ staticmethod
    def distributions_for_model(model_type: ModelType, prefix: str = "") -> Dict[str, Any]:
        """ Get the parameter distributions for the model.

        Parameters
        ----------
        model_type : ModelType
            The type of model.
        prefix : str, optional
            The prefix to add to the keys, by default "".

        Returns
        -------
        Dict[str, Any]
            The parameter distributions for the model.
        """
        return ParamDistributions.add_prefix_to_dict_keys(DISTRIBUTIONS_FOR_ALL_MODELS[model_type.value], prefix)

    @staticmethod
    def distributions_for_feature_extraction(feature_extraction_type: FeatureExtractorType, prefix: str = "") -> Dict[str, Any]:
        """ Get the parameter distributions for the feature extraction.

        Parameters
        ----------
        feature_extraction_type : FeatureExtractorType
            The type of feature extraction.
        prefix : str, optional
            The prefix to add to the keys, by default "".

        Returns
        -------
        Dict[str, Any]
            The parameter distributions for the feature extraction.
        """
        return ParamDistributions.add_prefix_to_dict_keys(FEATURE_EXTRACTION_PARAMS[feature_extraction_type.value], prefix)

    @staticmethod
    def distributions_for_pipeline(feature_extraction_type: FeatureExtractorType,
                                   model_type: ModelType,
                                   feature_pipeline_prefix: str,
                                   model_prefix: str
                                   ) -> Dict[str, Any]:
        """ Get the parameter distributions for the pipeline.

        Parameters
        ----------
        feature_extraction_type : FeatureExtractorType
            The type of feature extraction.
        model_type : ModelType
            The type of model.
        feature_pipeline_prefix : str
            The prefix for the feature extraction.
        model_prefix : str
            The prefix for the model.

        Returns
        -------
        Dict[str, Any]
            The parameter distributions for the pipeline.
        """
        return {**ParamDistributions.distributions_for_feature_extraction(feature_extraction_type, feature_pipeline_prefix),
                **ParamDistributions.distributions_for_model(model_type, model_prefix)}
