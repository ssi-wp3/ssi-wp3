from wordcloud import WordCloud
import pandas as pd
import os


def series_to_set(series: pd.Series) -> set:
    """Converts a pandas series to a set"""
    return set(series.unique().tolist())


def jaccard_index(set1, set2):
    """ Computes the Jaccard index between two sets

    The Jaccard Index measures similarity between finite sample sets, 
    and is defined as the size of the intersection divided by the 
    size of the union of the sample sets.

    The function will return a value between 0 and 1, where 0 means no overlap and 1 means complete overlap.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union


def dice_coefficient(set1, set2) -> float:
    """ Computes the Dice coefficient between two sets

    Similar to the Jaccard index, but uses twice the intersection size in the numerator. 
    It's defined as 2 * |X ∩ Y| / (|X| + |Y|). 

    It ranges from 0 (no overlap) to 1 (complete overlap).

    """
    intersection = len(set1.intersection(set2))
    return 2. * intersection / (len(set1) + len(set2))


def overlap_coefficient(set1, set2):
    """ Computes the overlap coefficient between two sets

    Defined as |X ∩ Y| / min(|X|, |Y|). 
    It ranges from 0 (no overlap) to 1 (complete overlap).

    """
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2))


def wordcloud_from_set(set1, filename: str):
    """ Creates a wordcloud from a set of words """
    return WordCloud().generate(' '.join(set1)).to_file(filename)


def compare_receipt_texts(receipt_texts_left: set, receipt_texts_right: set):
    """Compares two receipt texts"""
    # receipt_texts_left_set = series_to_set(receipt_texts_left)
    # receipt_texts_right_set = series_to_set(receipt_texts_right)
    intersection = receipt_texts_left.intersection(receipt_texts_right)
    union = receipt_texts_left.union(receipt_texts_right)
    left_difference = receipt_texts_left.difference(
        receipt_texts_right)
    right_difference = receipt_texts_right.difference(
        receipt_texts_left)
    return {
        "jaccard_index": jaccard_index(receipt_texts_left, receipt_texts_right),
        "dice_coefficient": dice_coefficient(receipt_texts_left, receipt_texts_right),
        "overlap_coefficient": overlap_coefficient(receipt_texts_left, receipt_texts_right),
        "left_set_length": len(receipt_texts_left),
        "right_set_length": len(receipt_texts_right),
        "intersection_length": len(intersection),
        "union_length": len(union),
        "left_difference_length": len(left_difference),
        "right_difference_length": len(right_difference),
    }


def compare_receipt_texts_per_year(dataframe: pd.DataFrame,
                                   year_column: str = "year",
                                   receipt_text_column: str = "receipt_text") -> pd.DataFrame:
    """Compares receipt texts per year"""
    receipt_texts_per_year = dataframe.groupby(
        year_column)[receipt_text_column].apply(series_to_set)
    return pd.DataFrame([compare_receipt_texts(receipt_texts_left, receipt_texts_right)
                         for receipt_texts_left, receipt_texts_right in zip(receipt_texts_per_year, receipt_texts_per_year[1:])
                         ])


def compare_receipt_texts_per_month(dataframe: pd.DataFrame,
                                    month_column: str = "month",
                                    receipt_text_column: str = "receipt_text") -> pd.DataFrame:
    """Compares receipt texts per month"""
    receipt_texts_per_month = dataframe.groupby(
        month_column)[receipt_text_column].apply(series_to_set)
    return pd.DataFrame([compare_receipt_texts(receipt_texts_left, receipt_texts_right)
                         for receipt_texts_left, receipt_texts_right in zip(receipt_texts_per_month, receipt_texts_per_month[1:])
                         ])


def analyze_supermarket_receipts(filename: str,
                                 supermarket_name: str,
                                 output_directory: str,
                                 year_column: str = "year",
                                 month_column: str = "month",
                                 receipt_text_column: str = "receipt_text"):
    supermarket_dataframe = pd.read_parquet(filename, engine="pyarrow")
    receipts_per_year = compare_receipt_texts_per_year(
        supermarket_dataframe, year_column, receipt_text_column)
    receipts_per_year.to_csv(
        os.path.join(output_directory, f"{supermarket_name}_receipts_per_year.csv"))

    receipts_per_month = compare_receipt_texts_per_month(
        supermarket_dataframe, month_column, receipt_text_column)
    receipts_per_month.to_csv(
        os.path.join(output_directory, f"{supermarket_name}_receipts_per_month.csv"))
