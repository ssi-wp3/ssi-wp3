from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def vector_similarity_per_group(dataframe: pd.DataFrame, features: np.array) -> pd.DataFrame:
    similarity_matrix = cosine_similarity(features)
    dataframe['similarity'] = similarity_matrix.mean(axis=1)
    return dataframe.groupby('coicop')['similarity'].agg(['mean', 'std'])

def tfidf_features(dataframe, stop_words, text_column):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(dataframe[text_column])

def tfidf_similarity(dataframe: pd.DataFrame, stop_words: str = 'dutch', text_column: str = "ean_name") -> pd.DataFrame:
    # Assuming df is your DataFrame, 'text' is the column with the receipt texts, and 'coicop' is the COICOP category
    X = tfidf_features(dataframe, stop_words, text_column)
    return cosine_similarity(X)
  
def tfidf_similarity_per_group(dataframe: pd.DataFrame, stop_words: str = 'dutch') -> pd.DataFrame:
    X = tfidf_features(dataframe, stop_words) 
    return vector_similarity_per_group(dataframe, X)

def similary_plot(df: pd.DataFrame, stop_words: str = 'dutch', text_column: str = "ean_name"):
    # Assuming similarity_df is the DataFrame returned by the tfidf_similarity function
    similarity_df = tfidf_similarity(df, stop_words, text_column)

    similarity_df.plot(kind='bar', y='mean', yerr='std', capsize=4)
    plt.title('Similarity Scores by COICOP Category')
    plt.xlabel('COICOP Category')
    plt.ylabel('Similarity Score')
    plt.show()

def count_vectorizer_features(dataframe, stop_words, text_column):
    vectorizer = CountVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(dataframe[text_column])

def count_vectorizer_similarity(dataframe: pd.DataFrame, stop_words: str = 'dutch', text_column: str = 'ean_name') -> pd.DataFrame:
    X = count_vectorizer_features(dataframe, stop_words, text_column)
    return cosine_similarity(X)

def count_vectorizer_similarity_per_group(dataframe: pd.DataFrame, stop_words: str = 'dutch', text_column: str = 'ean_name') -> pd.DataFrame:
    X = count_vectorizer_features(dataframe, stop_words, text_column)
    return vector_similarity_per_group(dataframe, X) 

def plot_similarity_heatmap(similarity_matrix: pd.DataFrame):
    # Assuming similarity_matrix is the cosine similarity matrix
    sns.heatmap(similarity_matrix, cmap='coolwarm')
    plt.title('Similarity Matrix')
    plt.show()