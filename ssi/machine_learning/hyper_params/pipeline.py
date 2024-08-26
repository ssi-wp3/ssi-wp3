from .hyper_params import FeatureExtractorType
from .sampler import create_sampler_for_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def feature_extractor_for_type(feature_extractor_type: FeatureExtractorType):
    if feature_extractor_type == FeatureExtractorType.count_vectorizer:
        return CountVectorizer()
    elif feature_extractor_type == FeatureExtractorType.tfidf_vectorizer:
        return TfidfVectorizer()
    elif feature_extractor_type == FeatureExtractorType.hashing_vectorizer:
        return HashingVectorizer()


def pipeline_with(feature_extractor_type: FeatureExtractorType,
                  model,
                  feature_extractor_prefix: str = 'vectorizer',
                  model_prefix: str = 'clf') -> Pipeline:
    return Pipeline([
        (feature_extractor_prefix, feature_extractor_for_type(feature_extractor_type)),
        (model_prefix, model)
    ])


def pipeline_and_sampler_for(feature_extractor_type: FeatureExtractorType,
                             model,
                             number_of_iterations: int,
                             random_state: int = 42):
    sampler = create_sampler_for_pipeline(
        feature_extractor_type, model, number_of_iterations, random_state)
    pipeline = pipeline_with(feature_extractor_type, model)
    return sampler, pipeline
