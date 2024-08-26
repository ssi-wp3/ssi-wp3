from .hyper_params import FeatureExtractorType
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def feature_extractor_for_type(feature_extractor_type: FeatureExtractorType):
    if feature_extractor_type == FeatureExtractorType.count_vectorizer:
        return CountVectorizer()
    elif feature_extractor_type == FeatureExtractorType.tfidf_vectorizer:
        return TfidfVectorizer()
    elif feature_extractor_type == FeatureExtractorType.hashing_vectorizer:
        return HashingVectorizer()


def pipeline(feature_extractor_type: FeatureExtractorType,
             model,
             feature_extractor_prefix: str = 'vectorizer',
             model_prefix: str = 'clf') -> Pipeline:
    return Pipeline([
        (feature_extractor_prefix, feature_extractor_for_type(feature_extractor_type)),
        (model_prefix, model)
    ])
