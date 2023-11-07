from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd


def tfidf_similarity(dataframe: pd.DataFrame, stop_words: str = 'dutch') -> pd.DataFrame:
    # Assuming df is your DataFrame, 'text' is the column with the receipt texts, and 'coicop' is the COICOP category
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(dataframe['text'])
    return cosine_similarity(X)
  
def tfidf_similarity_per_group(dataframe: pd.DataFrame, stop_words: str = 'dutch') -> pd.DataFrame:
    similarity_matrix = tfidf_similarity(dataframe, stop_words)
    dataframe['similarity'] = similarity_matrix.mean(axis=1)
    return dataframe.groupby('coicop')['similarity'].agg(['mean', 'std'])

def similary_plot(df: pd.DataFrame, stop_words: str = 'dutch'):
    # Assuming similarity_df is the DataFrame returned by the tfidf_similarity function
    similarity_df = tfidf_similarity(df, stop_words)

    similarity_df.plot(kind='bar', y='mean', yerr='std', capsize=4)
    plt.title('Similarity Scores by COICOP Category')
    plt.xlabel('COICOP Category')
    plt.ylabel('Similarity Score')
    plt.show()


def count_vectorizer_similarity(dataframe: pd.DataFrame, stop_words: str = 'dutch') -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(dataframe['text'])
    return cosine_similarity(X)

def count_vectorizer_similarity_per_group(dataframe: pd.DataFrame, stop_words: str = 'dutch') -> pd.DataFrame:
    similarity_matrix = count_vectorizer_similarity(dataframe, stop_words)
    dataframe['similarity'] = similarity_matrix.mean(axis=1)
    return dataframe.groupby('coicop')['similarity'].agg(['mean', 'std'])