from ssi.feature_extraction import FeatureExtractorFactory, FeatureExtractorType, SpacyFeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ssi.synthetic_data import generate_fake_revenue_data
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
                                   SpacyFeatureExtractor))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.spacy_nl_md),
                                   SpacyFeatureExtractor))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.spacy_nl_lg),
                                   SpacyFeatureExtractor))

    def test_add_feature_vectors(self):
        dataframe = generate_fake_revenue_data(100, 2018, 2021)
        factory = FeatureExtractorFactory()
        feature_df = factory.add_feature_vectors(
            dataframe, "coicop_name", "cv_features", FeatureExtractorType.count_vectorizer)
        self.assertTrue("cv_features" in feature_df.columns)
        self.assertEqual(len(feature_df["cv_features"][0]), 5000)
