from typing import List, Tuple
import itertools


def store_combinations(available_stores: List[str]) -> List[Tuple[str, str]]:
    """Create all combinations of two stores from a list of available stores.

    Parameters
    ----------
    available_stores : List[str]
        A list of available stores

    Returns
    -------
    List[Tuple[str, str]]
        A list of all combinations of two stores
    """
    return list(itertools.combinations(available_stores, 2))
