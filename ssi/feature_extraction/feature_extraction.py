from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import issparse
from enum import Enum
from typing import Dict, Optional, List
from ..files import get_feature_filename
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
import tqdm
import math
import os


class FeatureExtractorType(Enum):
    """This enum contains all the feature extractor types that can be used in the feature extraction pipeline.
    The following feature extractors are available:
    - test_extractor: A test feature extractor that returns a counter for each text
    - count_vectorizer: A vectorizer that counts the number of times a word appears in a text
    - tfidf_word: A vectorizer that uses the term frequency-inverse document frequency (TF-IDF) to weigh the importance of a word in a text
    - tfidf_char: A vectorizer that uses the term frequency-inverse document frequency (TF-IDF) to weigh the importance of a character in a text
    - tfidf_char34: A vectorizer that uses the term frequency-inverse document frequency (TF-IDF) to weigh the importance of a character in a text
    - count_char: A vectorizer that counts the number of times a character appears in a text
    - spacy_nl_sm: A feature extractor that uses the spacy library to extract features from a text, using the small spacy model
    - spacy_nl_md: A feature extractor that uses the spacy library to extract features from a text, using the medium spacy model
    - spacy_nl_lg: A feature extractor that uses the spacy library to extract features from a text, using the large spacy model
    - hf_all_mini_lm: A feature extractor that uses the HuggingFace SentenceTransformer library to extract features from a text, using the all-MiniLM-L6-v2 model
    - hf_labse: A feature extractor that uses the HuggingFace SentenceTransformer library to extract features from a text, using the LaBSE model
    """
    test_extractor = 'test_extractor'
    count_vectorizer = 'count_vectorizer'
    tfidf_word = 'tfidf_word'
    tfidf_char = 'tfidf_char'
    tfidf_char34 = 'tfidf_char34'
    count_char = 'count_char'
    spacy_nl_sm = 'spacy_nl_sm'
    spacy_nl_md = 'spacy_nl_md'
    spacy_nl_lg = 'spacy_nl_lg'
    hf_all_mini_lm = 'hf_all_mini_lm'
    hf_labse = 'hf_labse'


class TestFeatureExtractor:
    """ This class is a test feature extractor that returns a automatically incremented counter for each text.
    It is used for testing purposes.
    """

    def __init__(self) -> None:
        """ Constructor for the TestFeatureExtractor class.
        """
        self._counter = 0

    @property
    def counter(self):
        """ Returns the underlying counter.
        """
        return self._counter

    def fit_transform(self, data):
        """ This method uses the feature extractor to extract features from the data.
        As the feature extractor is a test feature extractor, it just returns a counter for each text.

        Parameters
        ----------
        data : List[str]
            The list of strings to extract features from.

        Returns
        -------
        List[np.array[float]]
            A list of lists containing the feature vectors for each input string.
        """
        vectors = []
        for text in data:
            vectors.append([self._counter, 0])
            self._counter += 1
        return vectors


class SpacyFeatureExtractor:
    """ This class is a wrapper around the Spacy library.
    It can uses spacy to tokenize the text data and return the word vectors.
    """

    def __init__(self, model_name):
        """ Constructor for the SpacyFeatureExtractor class.
        Parameters
        ----------
        model_name : str
            The name of the spacy model to use for feature extraction.
        """
        self.nlp = spacy.load(model_name)

    # To speed up: https://github.com/explosion/spaCy/discussions/8402
    def fit(self, X, y, **fit_params):
        """ This method is not used for this feature extractor.
        """
        pass

    def transform(self, X):
        """ This method uses the feature extractor to extract embeddings from the data.

        Parameters
        ----------
        X : List[str]
            The list of strings to extract features from.

        Returns
        -------
        List[np.array[float]]
            A list of lists containing the feature vectors for each input string.
        """
        # We only need spacy to tokenize the text and return the word vectors
        return [doc.vector
                for doc in self.nlp.pipe(X, disable=["tagger", "parser", "ner"])]

    def fit_transform(self, X, y=None, **fit_params):
        """ This method uses the feature extractor to extract embeddings from the data.
        As Spacy models are already pretrained, this method is equivalent to transform
        and just calls this method instead.

        Parameters
        ----------
        X : List[str]
            The list of strings to extract features from.

        Returns
        -------
        List[np.array[float]]
            A list of lists containing the feature vectors for each input string.
        """
        return self.transform(X)


class HuggingFaceFeatureExtractor:
    """ This class is a wrapper around the HuggingFace SentenceTransformer library.
    It uses the SentenceTransformer to encode the text data into feature vectors.
    All sentence transformers on HuggingFace can be used as feature extractors using this class.
    """

    def __init__(self, model_name: str, device: str = "cuda:0", batch_size: int = 1000):
        """ Constructor for the HuggingFaceFeatureExtractor class.

        Parameters
        ----------
        model_name : str
            The name of the model to use for feature extraction.

        device : str
            The device to use for feature extraction. Default is "cuda:0", or the first CUDA GPU.
        """
        self.__model = model_name
        self.__device = device
        self.__embedding_model = None

    @property
    def model(self):
        """ Returns the model name.

        Returns
        -------
        str
            The name of the model.
        """
        return self.__model

    @property
    def device(self):
        """ Returns the device. For instance, "cuda:0" or "cpu".

        Returns
        -------
        str
            The device.
        """
        return self.__device

    @property
    def embedding_model(self):
        """ Returns the embedding model used for feature extraction.

        Returns
        -------
        SentenceTransformer
            The embedding model.
        """
        if not self.__embedding_model:
            self.__embedding_model = SentenceTransformer(self.model)
            self.__embedding_model.to(self.device)
        return self.__embedding_model

    def fit(self, X, y, **fit_params):
        """ This method is not used for this feature extractor.
        """
        pass

    def transform(self, X):
        """ This method uses the feature extractor to extract embeddings from the data.

        Parameters
        ----------
        X : List[str]
            The list of strings to extract features from.

        Returns
        -------
        List[np.array[float]]
            A list of lists containing the feature vectors for each input string.
        """
        embedding = self.embedding_model.encode(
            X.values.tolist(), convert_to_tensor=True, batch_size=len(X))
        return embedding.cpu().detach().numpy()

    def fit_transform(self, X, y=None, **fit_params):
        """ This method uses the feature extractor to extract embeddings from the data.
        As HuggingFace models are already pretrained, this method is equivalent to transform
        and just calls this method instead.

        Parameters
        ----------
        X : List[str]
            The list of strings to extract features from.

        Returns
        -------
        List[np.array[float]]
            A list of lists containing the feature vectors for each input string.
        """
        return self.transform(X)


class FeatureExtractorFactory:
    """ This class is a factory for the feature extractors.
    It creates and returns the feature extractor for the given feature extractor type.
    """

    def __init__(self):
        """ Constructor for the FeatureExtractorFactory class.
        """
        self._feature_extractors = None

    @property
    def feature_extractors(self) -> Dict[FeatureExtractorType, object]:
        """ Returns the feature extractors.

        Returns
        -------
        Dict[FeatureExtractorType, object]
            A dictionary containing the feature extractors.
        """
        if not self._feature_extractors:
            self._feature_extractors = {
                FeatureExtractorType.test_extractor: TestFeatureExtractor(),

                # Sklearn feature extractors
                FeatureExtractorType.count_vectorizer: CountVectorizer(analyzer='word', token_pattern=r'\w{2,}', max_features=5000),
                FeatureExtractorType.tfidf_word: TfidfVectorizer(analyzer='word', token_pattern=r'\w{2,}', max_features=5000),
                FeatureExtractorType.tfidf_char: TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000),
                FeatureExtractorType.tfidf_char34: TfidfVectorizer(analyzer='char', ngram_range=(3, 4), max_features=5000),
                FeatureExtractorType.count_char: CountVectorizer(analyzer='char', max_features=5000),

                # Spacy feature extractors
                FeatureExtractorType.spacy_nl_sm: SpacyFeatureExtractor('nl_core_news_sm'),
                FeatureExtractorType.spacy_nl_md: SpacyFeatureExtractor('nl_core_news_md'),
                FeatureExtractorType.spacy_nl_lg: SpacyFeatureExtractor(
                    'nl_core_news_lg'),

                # HuggingFace feature extractors
                FeatureExtractorType.hf_all_mini_lm: HuggingFaceFeatureExtractor(
                    'sentence-transformers/all-MiniLM-L6-v2'),
                FeatureExtractorType.hf_labse: HuggingFaceFeatureExtractor(
                    'sentence-transformers/LaBSE')
            }
        return self._feature_extractors

    @property
    def feature_extractor_types(self):
        """ Returns the feature extractor types.

        Returns
        -------
        List[FeatureExtractorType]
            A list of feature extractor types.
        """
        return [feature_extractor_type
                for feature_extractor_type in self.feature_extractors.keys()
                if feature_extractor_type != FeatureExtractorType.test_extractor]

    def create_feature_extractor(self, feature_extractor_type: FeatureExtractorType):
        """ Creates and returns the feature extractor for the given feature extractor type.

        Parameters
        ----------
        feature_extractor_type : FeatureExtractorType
            The type of feature extractor to create.

        Returns
        -------
        object
            The feature extractor.
        """
        if feature_extractor_type not in self.feature_extractors:
            raise ValueError("Invalid type")
        return self.feature_extractors[feature_extractor_type]

    def add_feature_vectors(self,
                            dataframe: pd.DataFrame,
                            source_column: str,
                            destination_column: str,
                            feature_extractor_type: FeatureExtractorType,
                            filename: str,
                            batch_size: int,
                            progress_bar: Optional[tqdm.tqdm] = None
                            ):
        """ This method adds the feature vectors to the dataframe and saves it to a parquet file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to add the feature vectors to.

        source_column : str
            The column containing the text data to extract features from.

        destination_column : str
            The column to store the feature vectors in.

        filename : str
            The filename to save the feature vectors to.

        batch_size : int
            The number of rows to process at a time.

        progress_bar : Optional[tqdm.tqdm]
            The progress bar to update.
        """
        feature_extractor = self.create_feature_extractor(
            feature_extractor_type)

        pq_writer = None

        # Create directory if it does not exist
        if isinstance(filename, str):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        for i in range(0, len(dataframe), batch_size):
            if progress_bar:
                progress_bar.set_description(
                    f"Encoding batch {i // batch_size} out of {math.ceil(len(dataframe) / batch_size)} for {feature_extractor_type}")

            batch_df = dataframe.iloc[i:i+batch_size].copy()
            # Bit of a hack, should be fixed in a different way in the future
            batch_df = batch_df.fillna('')
            vectors = feature_extractor.fit_transform(
                batch_df[source_column])

            batch_df[destination_column] = list(
                vectors.toarray()) if issparse(vectors) else list(vectors)

            table = pa.Table.from_pandas(batch_df)
            if i == 0:
                pq_writer = pq.ParquetWriter(filename, table.schema)
            pq_writer.write_table(table)
            if not progress_bar:
                continue
            progress_bar.update(batch_size)

        if pq_writer:
            pq_writer.close()

    def extract_features_and_save(self,
                                  dataframe: pd.DataFrame,
                                  source_column: str,
                                  destination_column: str,
                                  filename: str,
                                  feature_extractor_type: FeatureExtractorType,
                                  batch_size: int = 1000,
                                  progress_bar: Optional[tqdm.tqdm] = None):
        """ This method extracts the feature vectors from the dataframe and saves them to a parquet file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to extract features from.

        source_column : str
            The column containing the text data to extract features from.

        destination_column : str
            The column to store the feature vectors in.

        filename : str
            The filename to save the feature vectors to.

        feature_extractor_type : FeatureExtractorType
            The type of feature extractor to use.

        batch_size : int
            The number of rows to process at a time.

        progress_bar : Optional[tqdm.tqdm]
            The progress bar to update.
        """
        self.add_feature_vectors(
            dataframe, source_column, destination_column, feature_extractor_type, filename=filename, batch_size=batch_size, progress_bar=progress_bar)

    def extract_all_features_and_save(self,
                                      dataframe: pd.DataFrame,
                                      source_column: str,
                                      supermarket_name: str,
                                      output_directory: str,
                                      feature_extractors: Optional[List[FeatureExtractorType]] = None,
                                      batch_size: int = 1000):
        """ This method extracts the feature vectors from the dataframe and saves them to a parquet file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to extract features from.

        source_column : str
            The column containing the text data to extract features from.

        output_directory : str
            The directory to save the feature vectors to.

        feature_extractors : Optional[List[FeatureExtractorType]]
            The list of feature extractor types to use. If not specified, all feature extractor types are used.

        batch_size : int
            The number of rows to process at a time.
        """
        feature_extractors = self.feature_extractor_types if not feature_extractors else feature_extractors
        with tqdm.tqdm(total=len(self.feature_extractor_types), desc="Extracting features", unit="type") as progress_bar:
            for feature_extractor_type in feature_extractors:
                feature_filename = os.path.join(output_directory, get_feature_filename(
                    feature_extractor_type.value, supermarket_name))
                progress_bar.set_description(
                    f"Extracting features of type {feature_extractor_type.value} to {feature_filename}")
                self.extract_features_and_save(dataframe,
                                               source_column,
                                               f"features_{feature_extractor_type.value}",
                                               feature_filename,
                                               feature_extractor_type,
                                               batch_size=batch_size,
                                               progress_bar=progress_bar)
                progress_bar.update()
