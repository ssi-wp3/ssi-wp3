
from typing import Tuple, Callable


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
    def overlap_function(left_set: set, right_set: set): return len(
        left_set.intersection(right_set)) / len(left_set.union(right_set))
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
