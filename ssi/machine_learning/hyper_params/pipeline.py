from .hyper_params import ModelType, FeatureExtractorType
from .sampler import create_sampler_for_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def feature_extractor_for_type(feature_extractor_type: FeatureExtractorType):
    """ Get the feature extractor for the type.

    Parameters
    ----------
    feature_extractor_type : FeatureExtractorType
        The type of feature extractor.

    Returns
    -------
    sklearn.feature_extraction.text.CountVectorizer | sklearn.feature_extraction.text.TfidfVectorizer | sklearn.feature_extraction.text.HashingVectorizer
        The feature extractor for the type.
    """
    if feature_extractor_type == FeatureExtractorType.count_vectorizer:
        return CountVectorizer()
    elif feature_extractor_type == FeatureExtractorType.tfidf_vectorizer:
        return TfidfVectorizer()
    elif feature_extractor_type == FeatureExtractorType.hashing_vectorizer:
        return HashingVectorizer()


def model_for_type(model_type: ModelType):
    """ Get the model for the type.

    Parameters
    ----------
    model_type : ModelType
        The type of model.

    Returns
    ------- 
    sklearn.linear_model.LogisticRegression
        The model for the type.
    """
    if model_type == ModelType.logistic_regression:
        return LogisticRegression()


def pipeline_with(feature_extractor_type: FeatureExtractorType,
                  model: ModelType,
                  feature_extractor_prefix: str = 'vectorizer',
                  model_prefix: str = 'clf') -> Pipeline:
    """ Get the pipeline with the feature extractor and model.

    Parameters
    ----------
    feature_extractor_type : FeatureExtractorType
        The type of feature extractor.
    model : ModelType
        The type of model.
    feature_extractor_prefix : str, optional
        The prefix for the feature extractor, by default 'vectorizer'.
    model_prefix : str, optional
        The prefix for the model, by default 'clf'.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The pipeline with the feature extractor and model.
    """
    return Pipeline([
        (feature_extractor_prefix, feature_extractor_for_type(feature_extractor_type)),
        (model_prefix, model_for_type(model))
    ])


def pipeline_and_sampler_for(feature_extractor_type: FeatureExtractorType,
                             model: ModelType,
                             number_of_iterations: int,
                             feature_extractor_prefix: str = 'vectorizer',
                             model_prefix: str = 'clf',
                             random_state: int = 42):
    """ Get the pipeline and sampler for the feature extractor and model.

    Parameters
    ----------
    feature_extractor_type : FeatureExtractorType
        The type of feature extractor.
    model : ModelType
        The type of model.
    number_of_iterations : int
        The number of iterations for the sampler.
    feature_extractor_prefix : str, optional
        The prefix for the feature extractor, by default 'vectorizer'.
    model_prefix : str, optional
        The prefix for the model, by default 'clf'.
    random_state : int, optional
        The random state for the sampler, by default 42.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        The sampler for the pipeline.
    sklearn.pipeline.Pipeline
        The pipeline with the feature extractor and model.
    """
    sampler = create_sampler_for_pipeline(
        feature_extractor_type, model, number_of_iterations, feature_extractor_prefix, model_prefix, random_state)
    pipeline = pipeline_with(feature_extractor_type,
                             model, feature_extractor_prefix, model_prefix)
    return sampler, pipeline
