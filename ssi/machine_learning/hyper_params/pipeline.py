from .hyper_params import ModelType, FeatureExtractorType
from .sampler import create_sampler_for_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def feature_extractor_for_type(feature_extractor_type: FeatureExtractorType):
    if feature_extractor_type == FeatureExtractorType.count_vectorizer:
        return CountVectorizer()
    elif feature_extractor_type == FeatureExtractorType.tfidf_vectorizer:
        return TfidfVectorizer()
    elif feature_extractor_type == FeatureExtractorType.hashing_vectorizer:
        return HashingVectorizer()


def model_for_type(model_type: ModelType):
    if model_type == ModelType.logistic_regression:
        return LogisticRegression()


def pipeline_with(feature_extractor_type: FeatureExtractorType,
                  model: ModelType,
                  feature_extractor_prefix: str = 'vectorizer',
                  model_prefix: str = 'clf') -> Pipeline:
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
    sampler = create_sampler_for_pipeline(
        feature_extractor_type, model, number_of_iterations, feature_extractor_prefix, model_prefix, random_state)
    pipeline = pipeline_with(feature_extractor_type,
                             model, feature_extractor_prefix, model_prefix)
    return sampler, pipeline
