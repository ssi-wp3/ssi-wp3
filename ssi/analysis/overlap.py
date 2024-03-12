
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
    intersection = len(left_set.intersection(right_set))
    union = len(left_set) + len(right_set) - intersection
    return intersection / union


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
    intersection = len(left_set.intersection(right_set))
    return 2. * intersection / (len(left_set) + len(right_set))


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
    intersection = len(left_set.intersection(right_set))
    return intersection / min(len(left_set), len(right_set))
