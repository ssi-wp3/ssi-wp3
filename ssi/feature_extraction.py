from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import issparse
from enum import Enum
from typing import Dict
from .files import get_feature_filename
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
import tqdm
import os


class FeatureExtractorType(Enum):
    count_vectorizer = 'count_vectorizer'
    tfidf_word = 'tfidf_word'
    tfidf_char = 'tfidf_char'
    tfidf_char34 = 'tfidf_char34'
    count_char = 'count_char'
    spacy_nl_sm = 'spacy_nl_sm'
    spacy_nl_md = 'spacy_nl_md'
    spacy_nl_lg = 'spacy_nl_lg'


class SpacyFeatureExtractor:
    def __init__(self, model_name):
        self.nlp = spacy.load(model_name)

    def fit_transform(self, data):
        return [self.nlp(text).vector for text in data]


class FeatureExtractorFactory:
    def __init__(self):
        self._feature_extractors = None

    @property
    def feature_extractors(self) -> Dict[FeatureExtractorType, object]:
        if not self._feature_extractors:
            self._feature_extractors = {
                FeatureExtractorType.count_vectorizer: CountVectorizer(analyzer='word', token_pattern=r'\w{2,}', max_features=5000),
                FeatureExtractorType.tfidf_word: TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', max_features=5000),
                FeatureExtractorType.tfidf_char: TfidfVectorizer(analyzer='char', token_pattern=r'\w{2,}', ngram_range=(2, 3), max_features=5000),
                FeatureExtractorType.tfidf_char34: TfidfVectorizer(analyzer='char', token_pattern=r'\w{2,}', ngram_range=(3, 4), max_features=5000),
                FeatureExtractorType.count_char: CountVectorizer(analyzer='char', token_pattern=r'\w{2,}', max_features=5000),
                FeatureExtractorType.spacy_nl_sm: SpacyFeatureExtractor('nl_core_news_sm'),
                FeatureExtractorType.spacy_nl_md: SpacyFeatureExtractor('nl_core_news_md'),
                FeatureExtractorType.spacy_nl_lg: SpacyFeatureExtractor(
                    'nl_core_news_lg')
            }
        return self._feature_extractors

    @property
    def feature_extractor_types(self):
        return self.feature_extractors.keys()

    def create_feature_extractor(self, feature_extractor_type: FeatureExtractorType):
        if feature_extractor_type in self.feature_extractors:
            return self.feature_extractors[feature_extractor_type]
        else:
            raise ValueError("Invalid type")

    def add_feature_vectors(self,
                            dataframe: pd.DataFrame,
                            source_column: str,
                            destination_column: str,
                            feature_extractor_type: FeatureExtractorType,
                            filename: str,
                            batch_size: int = 1000,
                            ):
        feature_extractor = self.create_feature_extractor(
            feature_extractor_type)

        pq_writer = None
        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i+batch_size]
            vectors = feature_extractor.fit_transform(batch[source_column])
            vectors_df = pd.DataFrame({destination_column: list(
                vectors.toarray()) if issparse(vectors) else list(vectors)})
            # Create directory if it does not exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            table = pa.Table.from_pandas(vectors_df)
            if i == 0:
                pq_writer = pq.ParquetWriter(filename, table.schema)
            pq_writer.write_table(table)

            # vectors_df.to_parquet(filename, engine='pyarrow', index=False,
            #                      append=True if i > 0 else False)
        if pq_writer:
            pq_writer.close()

    def extract_features_and_save(self, dataframe: pd.DataFrame, source_column: str, destination_column: str, filename: str, feature_extractor_type: FeatureExtractorType):
        self.add_feature_vectors(
            dataframe, source_column, destination_column, feature_extractor_type, filename=filename)

    def extract_all_features_and_save(self, dataframe: pd.DataFrame, source_column: str, supermarket_name: str, output_directory: str):
        with tqdm.tqdm(total=len(self.feature_extractor_types), desc="Extracting features", unit="type") as progress_bar:
            for feature_extractor_type in self.feature_extractor_types:
                feature_filename = os.path.join(output_directory, get_feature_filename(
                    feature_extractor_type.value, supermarket_name))
                progress_bar.set_description(
                    f"Extracting features of type {feature_extractor_type.value} to {feature_filename}")
                self.extract_features_and_save(dataframe,
                                               source_column, f"features_{feature_extractor_type.value}", feature_filename, feature_extractor_type)
                progress_bar.update()
