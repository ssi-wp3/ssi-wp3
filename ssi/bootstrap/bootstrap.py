from typing import Tuple, Dict, Any, Union, Optional, Callable, List
from sklearn.utils import resample
import pandas as pd


def bootstrap_sample(dataframe: pd.DataFrame,
                     n_samples: int = None,
                     replace: bool = True,
                     random_state: int = 42,
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use sklearn.utils.resample to return a bootstrapped sample of the input dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    replace : bool
        Whether to sample with replacement.

    n_samples : int
        The number of samples to draw. If None, it will be the same as the size of the input dataframe.

    random_state : int
        The random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The dataframe with the bootstrapped sample and a dataframe with the out of bag sample.

    """
    sample_indices = resample(
        dataframe.index, replace=replace, n_samples=n_samples, random_state=random_state)
    out_of_bag_indices = dataframe.index.difference(sample_indices)
    return dataframe.loc[sample_indices], dataframe.loc[out_of_bag_indices]


def bootstrap_sample_with_ratio(dataframe: pd.DataFrame,
                                sample_ratio: float = 1.0,
                                replace: bool = True,
                                random_state: int = 42,
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use sklearn.utils.resample to return a bootstrapped sample of the input dataframe.
    Instead of specifying the number of samples, the sample_ratio is used in this function to determine the number of samples. This function calculates the number of samples for the sample_ratio and calls the bootstrap_sample function.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    sample_ratio : float
        The ratio of samples to draw. If 1.0, it will be the same as the size of the input dataframe.

    replace : bool
        Whether to sample with replacement.

    random_state : int
        The random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The dataframe with the bootstrapped sample and a dataframe with the out of bag sample.

    """
    n_samples = int(len(dataframe) * sample_ratio)
    return bootstrap_sample(dataframe, n_samples=n_samples, replace=replace, random_state=random_state)


def perform_bootstrap(dataframe: pd.DataFrame,
                      n_bootstraps: int,
                      n_samples_per_bootstrap: Union[Optional[int], float],
                      evaluation_function: Callable[[int, int, pd.DataFrame, pd.DataFrame], Any],
                      replace: bool = True,
                      random_state: int = 42,
                      ) -> List[Dict[str, Any]]:
    """Perform bootstrapping on the input dataframe. The bootstrap will take n_bootstraps samples from the input dataframe and evaluate the evaluation_function on each of the bootstrapped samples.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    n_bootstraps : int
        The number of bootstraps to perform.

    n_samples_per_bootstrap : Union[Optional[int], float]
        The number of samples to draw for each bootstrap. If None, it will be the same as the size of the input dataframe. If an integer it will be the number of samples. If a float, it will be the ratio of samples to draw.

    evaluation_function : Callable[[int, int, pd.DataFrame, pd.DataFrame], Any]
        The function to evaluate on each bootstrapped sample. The function takes two integers and two dataframes as input and return a dictionary with evaluation metrics:
        - The first int is the number of the bootstrap currently being evaluated. 
        - The second int is the total number of bootstraps.
        - The first dataframe is the bootstrapped sample 
        - The second dataframe is the out of bag sample.

    replace : bool
        Whether to sample with replacement.

    random_state : int
        The random seed.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with the evaluation results for each bootstrap.
    """
    results = []
    for bootstrap_index in range(n_bootstraps):
        bootstrap_sample_df, out_of_bag_sample_df = bootstrap_sample_with_ratio(
            dataframe, sample_ratio=n_samples_per_bootstrap, replace=replace, random_state=random_state)
        results.append(evaluation_function(bootstrap_index, n_bootstraps,
                                           bootstrap_sample_df, out_of_bag_sample_df))
    return results
