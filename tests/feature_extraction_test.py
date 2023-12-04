from ssi.feature_extraction import FeatureExtractorFactory, FeatureExtractorType
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import unittest


class FeatureExtractionTest(unittest.TestCase):
    def test_create_feature_extractor(self):
        factory = FeatureExtractorFactory()
        self.assertTrue(isinstance(factory.create_feature_extractor(
            FeatureExtractorType.count_vectorizer), CountVectorizer))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.tfidf_word),
                                   TfidfVectorizer))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.tfidf_char),
                                   TfidfVectorizer))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.tfidf_char34),
                                   TfidfVectorizer))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.count_char),
                                   CountVectorizer))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.spacy_nl_sm),
                                   FeatureExtractorFactory.feature_extractors[FeatureExtractorType.spacy_nl_sm]))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.spacy_nl_md),
                                   FeatureExtractorFactory.feature_extractors[FeatureExtractorType.spacy_nl_md]))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.spacy_nl_lg),
                                   FeatureExtractorFactory.feature_extractors[FeatureExtractorType.spacy_nl_lg]))
