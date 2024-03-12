from typing import Tuple, Optional, List
from wordcloud import WordCloud
import pandas as pd
import os


def clean_text(text: pd.Series) -> pd.Series:
    """Cleans a text"""
    return text.str.replace('[^0-9a-zA-Z.,-/ ]', "", regex=True).str.lstrip().str.rstrip().str.lower()


def series_to_set(series: pd.Series, clean_text: bool = False) -> set:
    """Converts a pandas series to a set"""
    if clean_text:
        return set(clean_text(series.drop_duplicates()).tolist())
    return set(series.drop_duplicates().tolist())

# TODO does not work!


def dataframe_to_set(dataframe: pd.DataFrame, clean_text: bool = False) -> pd.DataFrame:
    """Converts a dataframe to a set"""
    return pd.DataFrame({
        column: [series_to_set(dataframe[column], clean_text=clean_text)]
        for column in dataframe.columns
    })


def string_length_histogram(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Returns a dataframe with a histogram of the string lengths of a column in a dataframe,
    sorted by string length. The index of the dataframe will be the string length and the column
    will be the count of strings with that length.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe to process

    column : str
        The column to return the string length histogram for.

    Returns
    -------

    pd.DataFrame
        A dataframe with a histogram of the string lengths of a column in a dataframe,
    """
    return dataframe[column].str.len().value_counts().sort_index().reset_index().rename(columns={"index": "string_length", column: "count"})


def wordcloud_from_words(words: List[str], filename: str, **kwargs):
    """ Creates a wordcloud from a list of words 

    Parameters
    ----------

    words : List[str]
        The words to create the wordcloud from

    filename : str
        The filename to save the wordcloud to

    **kwargs
        Additional arguments to pass to the WordCloud constructor
    """
    return WordCloud(**kwargs).generate(' '.join(words)).to_file(filename)


def wordcloud_from_words_and_frequencies(words: List[str], frequencies: List[int], filename: str, **kwargs):
    """ Creates a wordcloud from a list of words and their frequencies

    Parameters
    ----------

    words : List[str]
        The words to create the wordcloud from

    frequencies : List[int]
        The frequencies of the words

    filename : str
        The filename to save the wordcloud to

    **kwargs
        Additional arguments to pass to the WordCloud constructor

    """
    frequencies_dict = dict(zip(words, frequencies))
    return WordCloud(**kwargs).generate_from_frequencies(frequencies_dict).to_file(filename)


def split_words(dataframe: pd.DataFrame, text_column: str, delimiter: str = " ") -> pd.Series:
    """Splits the words in a text column and returns a series with the separate words for each text.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe to process

    text_column : str
        The column to split

    delimiter : str
        The delimiter to split the text column on

    Returns
    -------
    pd.Series
        A series with the separate words for each text.
    """
    return dataframe[text_column].str.split(delimiter, expand=False)


def unique_split_words(dataframe: pd.DataFrame, text_column: str, delimiter: str = " ") -> set:
    """Splits the words in a text column and returns a set with the unique words

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    text_column : str
        The column to split

    delimiter : str
        The delimiter to split the text column on

    Returns
    -------

    set
        A set with the unique words
    """
    return set(split_words(dataframe, text_column, delimiter).values.flatten().tolist())


def write_set_texts_to_file(set1, filename: str, delimiter=";", chunk_size: int = 80):
    """ Writes a set of texts to a file """
    with open(filename, "w") as text_file:
        sorted_set = sorted(set1)
        for i in range(0, len(set1), chunk_size):
            text_file.write(f'{delimiter}'.join(sorted_set[i:i+chunk_size]))
            text_file.write("\n")


def detect_product_differences(receipt_texts_before: set, receipt_texts_after: set) -> Tuple[set, set, set, set]:
    """Detects differences between two sets of texts

    Parameters
    ----------
    receipt_texts_before : set
        Set of receipt texts before, this is for example the set of receipt texts in a previous
        year or month

    receipt_texts_after : set
        Set of receipt texts after, this is for example the set of receipt texts in a current
        year or month

    Returns
    -------
    A tuple of four sets:
    - texts_kept_the_same: texts that are present in both sets
    - combined_texts: texts that are present in either set
    - texts_disappeared: texts that are present in the first set but not in the second set
    - new_texts: texts that are present in the second set but not in the first set
    """
    texts_kept_the_same = receipt_texts_before.intersection(
        receipt_texts_after)
    combined_texts = receipt_texts_before.union(receipt_texts_after)
    texts_disappeared = receipt_texts_before.difference(
        receipt_texts_after)
    new_texts = receipt_texts_after.difference(
        receipt_texts_before)

    return texts_kept_the_same, combined_texts, texts_disappeared, new_texts


def group_unique_values_per_period(dataframe: pd.DataFrame, period_column: str, value_column: str) -> pd.DataFrame:
    grouped_texts_per_month = dataframe.groupby(
        by=period_column)[value_column].apply(series_to_set)
    return grouped_texts_per_month.reset_index()


def get_unique_texts_and_eans_per_period(dataframe: pd.DataFrame, period_column: str, receipt_column: str, ean_column: str) -> pd.DataFrame:
    grouped_texts_per_month = group_unique_values_per_period(
        dataframe, period_column, receipt_column)
    grouped_eans_per_month = group_unique_values_per_period(
        dataframe, period_column, ean_column)
    return grouped_texts_per_month.merge(grouped_eans_per_month, on=period_column)


def compare_receipt_texts_per_period(dataframe: pd.DataFrame, period_column: str, receipt_text_column: str) -> pd.DataFrame:
    """Compares receipt texts per period"""
    receipt_texts_per_period = dataframe.groupby(
        period_column)[receipt_text_column].apply(series_to_set)

    combined_set = series_to_set(dataframe[receipt_text_column])
    for period in receipt_texts_per_period.index:
        # Detect which products disappeared and which products are new
        # Add this as a column to the dataframe
        period_texts = receipt_texts_per_period[period]
        new_texts = combined_set.difference(period_texts)

        new_texts_column = [True if text in new_texts else False
                            for text in dataframe[receipt_text_column]]

        dataframe[f"new_text_{period}"] = new_texts_column

    return dataframe
