
from typing import Dict, Tuple, Callable, List, Optional
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

    Defined as (|X ∩ Y| / |X| + |Y|) * 100.
    It ranges from 0 (no overlap) to 100 (complete overlap).

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


def asymmetrical_overlap(left_set: set, right_set: set) -> float:
    """ Computes the overlap between two sets,
        divided by set X. Intuitively, it describes
        the portion of the elements of X also found in
        Y.


    Defined as |X ∩ Y| / |X|.
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

    overlap = left_set.intersection(right_set)
    return len(overlap) / len(left_set)


def calculate_store_overlap(store_left: pd.DataFrame,
                            store_right: pd.DataFrame,
                            product_id_column: str,
                            overlap_function: Callable[[set, set], float],
                            preprocess_function: Callable[[pd.Series], pd.Series],
                            ):
    """Calculate the overlap between two stores when using a specific overlap function.

    Parameters
    ----------
    store_left: pd.DataFrame
        The first store to compare.

    store_right: pd.DataFrame
        The second store to compare.

    product_id_column: str
        The name of the column containing the product identifiers. Products can be identified by their EAN number
        or receipt text for example.

    overlap_function: Callable[[set, set], float]
        The overlap function to use to calculate the overlap between the stores.

    preprocess_function: Callable[[pd.Series], pd.Series]
        A function to preprocess the data before calculating the overlap. This function can be used to filter out
        duplicate products, or return tokens for the product texts. 

    Returns
    -------
    float
        The overlap between the two stores.
    """
    store1_set = set(preprocess_function(
        store_left[product_id_column]).values)
    store2_set = set(preprocess_function(
        store_right[product_id_column]).values)

    return overlap_function(store1_set, store2_set)


def calculate_overlap_for_stores(store_data: List[pd.DataFrame],
                                 store_id_column: str,
                                 product_id_column: str,
                                 overlap_function: Callable[[
                                     set, set], float] = jaccard_index,
                                 preprocess_function: Callable[[
                                     pd.Series], pd.Series] = lambda series: series,
                                 progress_bar: Optional[tqdm.tqdm] = None,
                                 calculate_all_cells: bool = False) -> pd.DataFrame:
    """ Calculate the overlap between the products of a list of stores.

    Parameters
    ----------
    store_data: List[pd.DataFrame]
        A list of dataframes, where each dataframe contains a column with the store name and a column with the store items.

    store_id_column: str
        The name of the column containing the store identifiers. Stores can be identified by their name or ID for example.

    product_id_column: str
        The name of the column containing the product identifiers. Products can be identified by their EAN number
        of receipt text for example.

    overlap_function: Callable[[set, set], float]
        The overlap function to use to calculate the overlap between the stores.

    preprocess_function: Callable[[pd.Series], pd.Series]
        A function to preprocess the data before calculating the overlap. This function can be used to filter out
        duplicate products, or return tokens for the product texts. By default, a lambda function is used that returns
        the series as is .

    progress_bar: Optional[tqdm.tqdm]
        A progress bar to show the progress of the calculation. By default, no progress bar is shown.

    calculate_all_cells: bool
        A flag to indicate if all cells in the overlap matrix should be calculated. By default, only the upper triangle
        of the matrix is calculated, as the matrix is symmetric.

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
        column_start_index = row_index if not calculate_all_cells else 0
        for column_index in range(column_start_index, len(store_data)):
            if progress_bar is not None:
                store_name1 = store_names[row_index]
                store_name2 = store_names[column_index]
                progress_bar.set_description(
                    f"Calculating overlap for store {store_name1} and store {store_name2}")

            store1 = store_data[row_index]
            store2 = store_data[column_index]
            overlap = calculate_store_overlap(
                store1,
                store2,
                product_id_column,
                overlap_function,
                preprocess_function)
            store_overlap[row_index, column_index] = overlap
            if not calculate_all_cells:
                store_overlap[column_index, row_index] = overlap

            if progress_bar is not None:
                progress_bar.update(1)

    return pd.DataFrame(store_overlap, columns=store_names, index=store_names)


def compare_overlap_between_preprocessing_functions(store_data: List[pd.DataFrame],
                                                    store_id_column: str,
                                                    product_id_column: str,
                                                    preprocess_functions: Dict[str, Callable[[pd.Series], pd.Series]],
                                                    overlap_function: Callable[[
                                                        set, set], float] = jaccard_index,
                                                    progress_bar: Optional[tqdm.tqdm] = None,
                                                    calculate_all_cells: bool = False
                                                    ) -> pd.DataFrame:
    """ Compare the overlap between stores for different preprocessing functions.

    Parameters
    ----------
    store_data: List[pd.DataFrame]
        A list of dataframes, where each dataframe contains a column with the store name and a column with the store items.

    store_id_column: str
        The name of the column containing the store identifiers. Stores can be identified by their name or ID for example.

    product_id_column: str
        The name of the column containing the product identifiers. Products can be identified by their EAN number
        or receipt text for example.

    preprocess_functions: Dict[str, Callable[[pd.Series], pd.Series]]
        A dictionary with the preprocessing functions to compare. The keys are the names of the preprocessing functions.

    overlap_function: Callable[[set, set], float]
        The overlap function to use to calculate the overlap between the stores.

    progress_bar: Optional[tqdm.tqdm]
        A progress bar to show the progress of the calculation. By default, no progress bar is shown.

    calculate_all_cells: bool
        A flag to indicate if all cells in the overlap matrix should be calculated. By default, only the upper triangle
        of the matrix is calculated, as the matrix is symmetric.

    Returns
    -------
    pd.DataFrame
        A dataframe with the mean overlap between the stores for each preprocessing function. The dataframe contains the
        mean overlap, the standard deviation, the minimum and maximum overlap for each preprocessing function.
    """
    mean_overlap_per_function = []
    for preprocess_function_name, preprocess_function in preprocess_functions.items():
        overlap_matrix = calculate_overlap_for_stores(
            store_data=store_data,
            store_id_column=store_id_column,
            product_id_column=product_id_column,
            overlap_function=overlap_function,
            preprocess_function=preprocess_function,
            progress_bar=None,
            calculate_all_cells=calculate_all_cells
        )
        progress_bar.set_description(
            f"Calculating mean overlap for preprocessing function {preprocess_function_name}")
        mean_overlap_per_function.append({
            "preprocessing_function": preprocess_function_name,
            "mean_overlap": overlap_matrix.values.mean(),
            "std_overlap": overlap_matrix.values.std(),
            "min_overlap": overlap_matrix.values.min(),
            "max_overlap": overlap_matrix.values.max()
        })
        progress_bar.update(1)
    return pd.DataFrame(mean_overlap_per_function)


def compare_overlap_per_coicop_label(store_data: List[pd.DataFrame],
                                     store_id_column: str,
                                     product_id_column: str,
                                     coicop_column: str,
                                     preprocess_function: Callable[[pd.Series], pd.Series],
                                     overlap_function: Callable[[
                                         set, set], float] = jaccard_index,
                                     progress_bar: Optional[tqdm.tqdm] = None) -> pd.DataFrame:
    """ Compare the overlap between stores for different preprocessing functions per COICOP label.
    The stores are compared pairwise for each COICOP label. By default, the Jaccard index is used to calculate the overlap.

    Parameters
    ----------
    store_data: List[pd.DataFrame]
        A list of dataframes, where each dataframe contains a column with the store name and a column with the store items.

    store_id_column: str
        The name of the column containing the store identifiers. Stores can be identified by their name or ID for example.

    product_id_column: str
        The name of the column containing the product identifiers. Products can be identified by their EAN number
        or receipt text for example.

    coicop_column: str
        The name of the column containing the COICOP labels.

    preprocess_function: Dict[str, Callable[[pd.Series], pd.Series]]
        The preprocessing function to use to preprocess the data before calculating the overlap.

    overlap_function: Callable[[set, set], float]
        The overlap function to use to calculate the overlap between the stores. Note that this function assumes that
        the overlap function is symmetric, as only the overlap between store A and store B is compared and the
        overlap between store B and store A is skipped.

    progress_bar: Optional[tqdm.tqdm]
        A progress bar to show the progress of the calculation. By default, no progress bar is shown.

    Returns
    -------
    pd.DataFrame
        A dataframe with the overlap between the stores for each preprocessing function per COICOP label.
    """
    all_overlap_dfs = []
    for store_left_index in range(len(store_data)):
        store_left = store_data[store_left_index]
        store_left_name = store_left[store_id_column].values[0]
        for store_right_index in range(store_left_index+1, len(store_data)):
            store_right = store_data[store_right_index]
            store_right_name = store_right[store_id_column].values[0]

            coicop_levels = set(store_left[coicop_column].unique()) | set(
                store_right[coicop_column].unique())

            overlap_values = []
            for coicop_label in coicop_levels:
                store_left_coicop = store_left[store_left[coicop_column]
                                               == coicop_label]
                store_right_coicop = store_right[store_right[coicop_column]
                                                 == coicop_label]

                overlap = calculate_store_overlap(
                    store_left_coicop,
                    store_right_coicop,
                    product_id_column,
                    overlap_function,
                    preprocess_function
                )
                overlap_values.append(overlap)
                progress_bar.set_description(
                    f"Calculating overlap for preprocessing function {preprocess_function.__name__} between store {store_left_name} and store {store_right_name} and coicop {coicop_label}")
                progress_bar.update(1)

            overlap_df = pd.DataFrame(
                overlap_values, columns=[f"{store_left_name}_{store_right_name}"], index=coicop_levels)
            all_overlap_dfs.append(overlap_df)

    return pd.concat(all_overlap_dfs, axis=1)
