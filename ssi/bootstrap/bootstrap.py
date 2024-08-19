from typing import Tuple, Dict, Any, Union, Optional, Callable, List
from sklearn.utils import resample
from collections import namedtuple
from sklearn.model_selection import train_test_split
import pandas as pd

BootstrapSample = namedtuple(
    "BootstrapSample", ["bootstrap_sample", "out_of_bag_sample"])


def bootstrap_sample(dataframe: pd.DataFrame,
                     n_samples: int = None,
                     replace: bool = True,
                     random_state: int = 42,
                     ) -> BootstrapSample:
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
    return BootstrapSample(dataframe.loc[sample_indices], dataframe.loc[out_of_bag_indices])


def bootstrap_sample_with_ratio(dataframe: pd.DataFrame,
                                sample_ratio: float = 1.0,
                                replace: bool = True,
                                random_state: int = 42,
                                ) -> BootstrapSample:
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
                      evaluation_function: Callable[[int, int, BootstrapSample, Optional[Dict[str, Any]]], Any],
                      preprocess_function: Optional[Callable[[
                          BootstrapSample, Optional[Dict[str, Any]]], BootstrapSample]] = None,
                      replace: bool = True,
                      random_state: int = 42,
                      **kwargs: Dict[str, Any]
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

    evaluation_function : Callable[[int, int, BootstrapSample], Any]
        The function to evaluate on each bootstrapped sample. The function takes two integers and two dataframes as input and return a dictionary with evaluation metrics:
        - The first int is the number of the bootstrap currently being evaluated.
        - The second int is the total number of bootstraps.
        - A namedtuple called BootStrapSample:
            - The first dataframe is the bootstrapped sample
            - The second dataframe is the out of bag sample.

    preprocess_function : Optional[Callable[[
        BootstrapSample], BootstrapSample]]
        A function to preprocess the bootstrapped sample before evaluating the evaluation_function.

    replace : bool
        Whether to sample with replacement.

    random_state : int
        The random seed.

    kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the evaluation function.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with the evaluation results for each bootstrap.
    """
    results = []
    for bootstrap_index in range(n_bootstraps):

        bootstrap_sample = bootstrap_sample_with_ratio(
            dataframe, sample_ratio=n_samples_per_bootstrap, replace=replace, random_state=random_state, **kwargs)
        if preprocess_function is not None:
            bootstrap_sample = preprocess_function(bootstrap_sample)

        results.append(evaluation_function(bootstrap_index, n_bootstraps,
                                           bootstrap_sample, **kwargs))
    return results


def perform_bootstrap_with_train_test_split(dataframe: pd.DataFrame,
                                            n_bootstraps: int,
                                            n_samples_per_bootstrap: Union[Optional[int], float],
                                            evaluation_function: Callable[[int, int, BootstrapSample], Any],
                                            preprocess_function: Optional[Callable[[
                                                BootstrapSample], BootstrapSample]] = None,
                                            replace: bool = True,
                                            test_size: float = 0.2,
                                            random_state: int = 42, **kwargs: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Perform bootstrapping on a dataset after performing a train-test split. The bootstrap will be performed on the training set and the final evaluation will be done on a hold out test set. The evaluation_function will be called on each bootstrapped sample and the results will be returned.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    n_bootstraps : int
        The number of bootstraps to perform.

    n_samples_per_bootstrap : Union[Optional[int], float]
        The number of samples to draw for each bootstrap. If None, it will be the same as the size of the input dataframe. If an integer it will be the number of samples. If a float, it will be the ratio of samples to draw.

    evaluation_function : Callable[[int, int, BootstrapSample], Any]
        The function to evaluate on each bootstrapped sample. The function takes two integers and two dataframes as input and return a dictionary with evaluation metrics:
        - The first int is the number of the bootstrap currently being evaluated.
        - The second int is the total number of bootstraps.
        - A namedtuple called BootStrapSample:
            - The first dataframe is the bootstrapped sample
            - The second dataframe is the out of bag sample.

    preprocess_function : Optional[Callable[[
        BootstrapSample], BootstrapSample]]
        A function to preprocess the bootstrapped sample before evaluating the evaluation_function.

    replace : bool
        Whether to sample with replacement.

    test_size : float
        The ratio of the test set size.

    random_state : int
        The random seed.

    kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the evaluation function.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[str, Any]]
        A tuple of with the evaluation results. The first paramater, a list, contains the evaluation results for the bootstraps on training set and the second parameter, a dict, contains the evaluation results for the test set.
    """
    training_df, test_df = train_test_split(dataframe, test_size=test_size)
    training_results = perform_bootstrap(training_df, n_bootstraps, n_samples_per_bootstrap,
                                         evaluation_function, preprocess_function, replace, random_state)

    # TODO which model?
    test_bootstrap_sample = BootstrapSample(test_df, pd.DataFrame())

    return
