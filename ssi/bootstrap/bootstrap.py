from sklearn.utils import resample
import pandas as pd


def bootstrap_sample(dataframe: pd.DataFrame,
                     replace: bool = True,
                     n_samples: int = None,
                     random_state: int = 42,
                     ) -> [pd.DataFrame, pd.DataFrame]:
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
    pd.DataFrame, pd.DataFrame
        The dataframe with the bootstrapped sample and a dataframe with the out of bag sample.

    """
    sample_indices = resample(
        dataframe.index, replace=replace, n_samples=n_samples, random_state=random_state)
    out_of_bag_indices = dataframe.index.difference(sample_indices)
    return dataframe.loc[sample_indices], dataframe.loc[out_of_bag_indices]
