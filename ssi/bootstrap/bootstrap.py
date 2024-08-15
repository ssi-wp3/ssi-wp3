from typing import Tuple
from sklearn.utils import resample
import pandas as pd


def bootstrap_sample(dataframe: pd.DataFrame,
                     replace: bool = True,
                     n_samples: int = None,
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
    return bootstrap_sample(dataframe, replace=replace, n_samples=n_samples, random_state=random_state)
