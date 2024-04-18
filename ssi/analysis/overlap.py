
from typing import Tuple, Callable, List, Optional
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import tqdm


def handle_missing_sets(left_set: set, right_set: set) -> Tuple[set, set]:
    """Handle missing sets

    Parameters
    ----------
    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    -------
    Tuple[set, set]
        A tuple with the two sets, where the missing set is replaced with an empty set.
    """
    left_set = set() if not left_set else left_set
    right_set = set() if not right_set else right_set
    return left_set, right_set


def __handle_zero_length_sets(left_set: set,
                              right_set: set,
                              overlap_function: Callable[[set, set], float],
                              default_exact_match: float = 1.0,
                              default_empty_match: float = 0.0,
                              ) -> Tuple[set, set]:
    """Handle zero length sets

    Parameters
    ----------
    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    default_value : float
        The default value to return if both sets are empty

    overlap_function : Callable[[set, set], float]
        The overlap function to use if both sets are not empty

    Returns
    -------
    Tuple[set, set]
        A tuple with the two sets, where the missing set is replaced with an empty set.
    """
    left_set, right_set = handle_missing_sets(left_set, right_set)
    if len(left_set) == 0 and len(right_set) == 0:
        return default_exact_match

    if left_set == right_set:
        return default_exact_match

    if len(left_set) == 0 or len(right_set) == 0:
        return default_empty_match
    return overlap_function(left_set, right_set)


def jaccard_similarity(left_set: set, right_set: set) -> float:
    """ Computes the Jaccard similarity between two sets, if both sets are empty, the function will return 1.0
    as both sets are equal.

    The Jaccard similarity measures similarity between finite sample sets,

    Parameters
    ----------
    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    -------
    float: The function will return a value between 0 and 1, where 0 means no overlap and 1 means complete overlap.
    """
    def overlap_function(left_set: set, right_set: set):
        return len(left_set.intersection(right_set)) / len(left_set.union(right_set))
    return __handle_zero_length_sets(left_set, right_set,
                                     overlap_function=overlap_function
                                     )


def jaccard_index(left_set: set, right_set: set) -> float:
    """ Computes the Jaccard index between two sets

    The Jaccard Index measures similarity between finite sample sets, 
    and is defined as the size of the intersection divided by the 
    size of the union of the sample sets.

    Parameters
    ----------
    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    ------- 
    float: The function will return a value between 0 and 1, where 0 means no overlap and 1 means complete overlap.

    """
    def overlap_function(left_set: set, right_set: set):
        intersection = len(left_set.intersection(right_set))
        union = len(left_set) + len(right_set) - intersection
        return intersection / union
    return __handle_zero_length_sets(left_set, right_set,
                                     overlap_function=overlap_function)


def dice_coefficient(left_set: set, right_set: set) -> float:
    """ Computes the Dice coefficient between two sets

    Similar to the Jaccard index, but uses twice the intersection size in the numerator. 
    It's defined as 2 * |X ∩ Y| / (|X| + |Y|). 

    It ranges from 0 (no overlap) to 1 (complete overlap).

    Parameters
    ----------
    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    -------

    float: The dice coefficient between the two sets.  
    """
    def overlap_function(left_set: set, right_set: set):
        intersection = len(left_set.intersection(right_set))
        return 2. * intersection / (len(left_set) + len(right_set))
    return __handle_zero_length_sets(left_set, right_set,
                                     overlap_function=overlap_function)


def overlap_coefficient(left_set: set, right_set: set) -> float:
    """ Computes the overlap coefficient between two sets

    Defined as |X ∩ Y| / min(|X|, |Y|). 
    It ranges from 0 (no overlap) to 1 (complete overlap).

    Parameters
    ----------

    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    -------
    float: The overlap coefficient between the two sets.
    """
    def overlap_function(left_set: set, right_set: set):
        intersection = len(left_set.intersection(right_set))
        min_length = min(len(left_set), len(right_set))
        return intersection / min_length
    return __handle_zero_length_sets(left_set, right_set,
                                     overlap_function=overlap_function)


def percentage_overlap(left_set: set, right_set: set) -> float:
    """ Computes the percentage overlap between two sets

    Defined as |X ∩ Y| / |X| + |Y|. 
    It ranges from 0 (no overlap) to 1 (complete overlap).

    Parameters
    ----------

    left_set : set
        The first set to compare

    right_set : set
        The second set to compare

    Returns
    -------
    float: The percentage overlap between the two sets.
    """
    def overlap_function(left_set: set, right_set: set):
        intersection = len(left_set.intersection(right_set))
        return (intersection / (len(left_set) + len(right_set)))
    return __handle_zero_length_sets(left_set, right_set,
                                     overlap_function=overlap_function) * 100


def split_strings(string_column: pd.Series, separator: str = ' ') -> pd.Series:
    """ Split strings in a column into separate words.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to split.

    separator : str
        The separator to use to split the strings. By default, the separator is a space.

    Returns
    -------
    pd.Series
        A series with the unique split strings.
    """
    return string_column.str.split(separator).explode()


def tokenize_strings(string_column: pd.Series, tokenizer: Callable[[str], List[str]]) -> pd.Series:
    """ Tokenize strings in a column using a custom tokenizer.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to tokenize.

    tokenizer : Callable[[str], List[str]]
        A function that takes a string as input and returns a list of tokens.

    Returns
    -------
    pd.Series
        A series with the unique tokens.
    """
    return string_column.apply(tokenizer).explode()


def huggingface_tokenizer(string_column: pd.Series, tokenizer_name: str = "gpt2") -> pd.Series:
    """ Tokenize strings in a column using a Hugging Face tokenizer.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to tokenize.

    tokenizer_name : str
        The name of the Hugging Face tokenizer to use. By default, the GPT-2 tokenizer is used.

    Returns
    -------
    pd.Series
        A series with the unique tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenize_strings(string_column, tokenizer.tokenize)


def calculate_overlap_for_stores(store_data: List[pd.DataFrame],
                                 store_id_column: str,
                                 product_id_column: str,
                                 overlap_function: Callable[[
                                     set, set], float] = jaccard_index,
                                 preprocess_function: Callable[[
                                     pd.Series], pd.Series] = lambda series: series,
                                 progress_bar: Optional[tqdm.tqdm] = None) -> pd.DataFrame:
    """ Calculate the overlap between the products of a list of stores.

    Parameters
    ----------
    store_data : List[pd.DataFrame]
        A list of dataframes, where each dataframe contains a column with the store name and a column with the store items.

    store_id_column : str
        The name of the column containing the store identifiers. Stores can be identified by their name or ID for example.

    product_id_column : str
        The name of the column containing the product identifiers. Products can be identified by their EAN number
        of receipt text for example.    

    overlap_function : Callable[[set, set], float]
        The overlap function to use to calculate the overlap between the stores.

    preprocess_function : Callable[[pd.Series], pd.Series]
        A function to preprocess the data before calculating the overlap. This function can be used to filter out
        duplicate products, or return tokens for the product texts. By default, a lambda function is used that returns
        the series as is.

    Returns
    -------
    pd.DataFrame
        A dataframe with the overlap between the stores. The dataframe contains a matrix where the rows and columns
        represent the store names and the values represent the overlap between the stores. The overlap matrix is
        symmetric, so the overlap between store A and store B is the same as the overlap between store B and store A.
    """
    number_of_stores = len(store_data)
    store_names = [store[store_id_column].values[0] for store in store_data]
    store_overlap = np.empty(
        (number_of_stores, number_of_stores), dtype=np.float64)
    for row_index in range(len(store_data)):
        for column_index in range(row_index, len(store_data)):
            if progress_bar is not None:
                store_name1 = store_names[row_index]
                store_name2 = store_names[column_index]
                progress_bar.set_description(
                    f"Calculating overlap for store {store_name1} and store {store_name2}")

            store1 = store_data[row_index]
            store2 = store_data[column_index]

            store1_set = set(preprocess_function(
                store1[product_id_column]).values)
            store2_set = set(preprocess_function(
                store2[product_id_column]).values)
            overlap = overlap_function(
                store1_set, store2_set)
            store_overlap[row_index, column_index] = overlap
            store_overlap[column_index, row_index] = overlap

            if progress_bar is not None:
                progress_bar.update(1)

    return pd.DataFrame(store_overlap, columns=store_names, index=store_names)
