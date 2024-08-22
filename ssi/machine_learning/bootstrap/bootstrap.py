from typing import Tuple, Dict, Any, Union, Optional, Callable, List
from sklearn.utils import resample
from collections import namedtuple
import pandas as pd
import csv

BootstrapSample = namedtuple(
    "BootstrapSample", ["bootstrap_sample", "oob_sample"])


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
    BootstrapSample
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
    BootstrapSample
        The dataframe with the bootstrapped sample and a dataframe with the out of bag sample.

    """
    n_samples = int(len(dataframe) * sample_ratio)
    return bootstrap_sample(dataframe, n_samples=n_samples, replace=replace, random_state=random_state)


def bootstrap(dataframe: pd.DataFrame,
              sample_ratio: Union[Optional[int], float],
              replace: bool = True,
              random_state: int = 42) -> BootstrapSample:
    """Bootstrap the input dataframe. This function is a wrapper around the bootstrap_sample and bootstrap_sample_with_ratio functions. If the sample_ratio is an integer, it will call the bootstrap_sample function. If the sample_ratio is a float, it will call the bootstrap_sample_with_ratio function.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    sample_ratio : Union[Optional[int], float]
        The number of samples to draw for each bootstrap. If None, it will be the same as the size of the input dataframe. If an integer it will be the number of samples. If a float, it will be the ratio of samples to draw.

    replace : bool
        Whether to sample with replacement.

    random_state : int
        The random seed.

    Returns
    -------
    BootstrapSample
        The dataframe with the bootstrapped sample and a dataframe with the out of bag sample.
    """
    if sample_ratio is None or isinstance(sample_ratio, int):
        return bootstrap_sample(dataframe, n_samples=sample_ratio, replace=replace, random_state=random_state)
    return bootstrap_sample_with_ratio(dataframe, sample_ratio=sample_ratio, replace=replace, random_state=random_state)


def perform_bootstrap(dataframe: pd.DataFrame,
                      n_bootstraps: int,
                      n_samples_per_bootstrap: Union[Optional[int], float],
                      results_file,
                      evaluation_function: Callable[[int, int, BootstrapSample, Optional[Dict[str, Any]]], Any],
                      preprocess_function: Optional[Callable[[
                          BootstrapSample, Optional[Dict[str, Any]]], BootstrapSample]] = None,
                      replace: bool = True,
                      random_state: int = 42,
                      **kwargs: Dict[str, Any]
                      ) -> pd.DataFrame:
    """Perform bootstrapping on the input dataframe. The bootstrap will take n_bootstraps samples from the input dataframe and evaluate the evaluation_function on each of the bootstrapped samples. This function trains the
    model on the bootstrapped sample and evaluates it on the out of bag sample. This gives an estimate of the model's performance on unseen data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    n_bootstraps : int
        The number of bootstraps to perform.

    n_samples_per_bootstrap : Union[Optional[int], float]
        The number of samples to draw for each bootstrap. If None, it will be the same as the size of the input dataframe. If an integer it will be the number of samples. If a float, it will be the ratio of samples to draw.

    results_file
        The file (already opened) to save the results.

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
    pd.DataFrame
        A dataframe with the evaluation metrics for each bootstrap.
    """
    # results = []

    for bootstrap_index in range(n_bootstraps):

        bootstrap_sample = bootstrap(
            dataframe, sample_ratio=n_samples_per_bootstrap, replace=replace, random_state=random_state)
        if preprocess_function is not None:
            bootstrap_sample = preprocess_function(
                bootstrap_sample, **kwargs)

        evaluation_dict = evaluation_function(bootstrap_index, n_bootstraps,
                                              bootstrap_sample, **kwargs)
        print(evaluation_dict)
        if bootstrap_index == 0:
            csv_writer = csv.DictWriter(
                results_file, fieldnames=evaluation_dict.keys())
            csv_writer.writeheader()
        csv_writer.writerow(evaluation_dict)
        results_file.flush()

    # return pd.DataFrame(results)


def perform_separate_bootstraps(dataframe: pd.DataFrame,
                                n_bootstraps: int,
                                n_samples_per_bootstrap: Union[Optional[int], float],
                                evaluation_function: Callable[[int, int, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]], Any],
                                preprocess_function: Optional[Callable[[
                                    pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]], BootstrapSample]] = None,
                                replace: bool = True,
                                random_state: int = 42,
                                **kwargs: Dict[str, Any]
                                ) -> pd.DataFrame:
    """Perform bootstrapping on the input dataframe. The bootstrap will take n_bootstraps samples from the input dataframe and evaluate the evaluation_function on each of the bootstrapped samples. This function takes two samples: one for the training data and one for the test data. Both samples are taken from the complete input
    dataframe and if replace is True with replacement. This function evaluates the model on test data that has an
    overlap with the training data. This is useful to evaluate the model in a more realistic scenario where some
    products are seen in the training data and the test data, i.e. some products will be available in the store's
    inventory in larger periods of time.

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
        - The first dataframe is the bootstrapped sample.
        - The second dataframe is the out of bag sample.

    preprocess_function : Optional[Callable[[
        pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]]
        A function to preprocess the bootstrapped sample before evaluating the evaluation_function.

    replace : bool
        Whether to sample with replacement.

    random_state : int
        The random seed.

    kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the evaluation function.

    Returns
    -------
    pd.DataFrame
        A list of dictionaries with the evaluation results for each bootstrap.
    """
    results = []
    for bootstrap_index in range(n_bootstraps):
        training_sample = bootstrap(
            dataframe, sample_ratio=n_samples_per_bootstrap, replace=replace, random_state=random_state)
        if preprocess_function is not None:
            training_sample = preprocess_function(training_sample, **kwargs)
        training_sample = training_sample.bootstrap_sample

        test_sample = bootstrap(
            dataframe, sample_ratio=n_samples_per_bootstrap, replace=replace, random_state=random_state)
        if preprocess_function is not None:
            test_sample = preprocess_function(test_sample, **kwargs)
        test_sample = test_sample.bootstrap_sample

        train_test_sample = BootstrapSample(training_sample, test_sample)
        results.append(evaluation_function(bootstrap_index, n_bootstraps,
                                           train_test_sample.bootstrap_sample, train_test_sample.out_of_bag_sample, **kwargs))
    return pd.DataFrame(results)
