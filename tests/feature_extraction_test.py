from ssi.feature_extraction.feature_extraction import FeatureExtractorFactory, FeatureExtractorType, SpacyFeatureExtractor, HuggingFaceFeatureExtractor
from ssi.synthetic_data import generate_fake_revenue_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from test_utils import get_test_path
import unittest
import pandas as pd
import os


class FeatureExtractionTest(unittest.TestCase):
    def test_feature_extractor_type_to_string(self):
        self.assertEqual("count_vectorizer",
                         f"{FeatureExtractorType.count_vectorizer.value}")

    def test_feature_extractor_types_property_returns_all_but_test_extractor(self):
        factory = FeatureExtractorFactory()
        self.assertEqual(10, len(factory.feature_extractor_types))
        self.assertFalse(
            FeatureExtractorType.test_extractor in factory.feature_extractor_types)
        self.assertEqual(set([FeatureExtractorType.count_vectorizer,
                              FeatureExtractorType.tfidf_word,
                              FeatureExtractorType.tfidf_char,
                              FeatureExtractorType.tfidf_char34,
                              FeatureExtractorType.count_char,
                              FeatureExtractorType.spacy_nl_sm,
                              FeatureExtractorType.spacy_nl_md,
                              FeatureExtractorType.spacy_nl_lg,
                              FeatureExtractorType.hf_all_mini_lm,
                              FeatureExtractorType.hf_labse]),
                         set(factory.feature_extractor_types))

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
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.hf_all_mini_lm),
                                   HuggingFaceFeatureExtractor))
        self.assertTrue(isinstance(factory.create_feature_extractor(FeatureExtractorType.hf_labse),
                                   HuggingFaceFeatureExtractor))

    def test_add_feature_vectors(self):
        dataframe = generate_fake_revenue_data(100, 2018, 2021)
        factory = FeatureExtractorFactory()

        features_filename = get_test_path("feature_test.parquet")
        if os.path.exists(features_filename):
            os.remove(features_filename)

        factory.add_feature_vectors(dataframe,
                                    source_column="coicop_name",
                                    destination_column="cv_features",
                                    filename=features_filename,
                                    batch_size=10,
                                    feature_extractor_type=FeatureExtractorType.count_vectorizer)

        feature_df = pd.read_parquet(features_filename, engine="pyarrow")
        self.assertTrue("cv_features" in feature_df.columns)
        self.assertEqual(100, len(feature_df))

    def test_extract_features_and_save(self):
        dataframe = generate_fake_revenue_data(100, 2018, 2021)
        factory = FeatureExtractorFactory()

        test_path = get_test_path("test.parquet")

        factory.extract_features_and_save(
            dataframe, "coicop_name", "cv_features", test_path, FeatureExtractorType.test_extractor)

        feature_df = pd.read_parquet(test_path, engine="pyarrow")

        expected_feature_df = dataframe.copy()
        expected_feature_df["cv_features"] = [[i, 0] for i in range(0, 100)]

        self.assertEqual(dataframe.columns.tolist() +
                         ["cv_features"], feature_df.columns.tolist())
        self.assertEqual(100, len(feature_df))
        for column in dataframe.columns:
            self.assertTrue(expected_feature_df[column].equals(
                feature_df[column]), f"Column {column} is not equal")
