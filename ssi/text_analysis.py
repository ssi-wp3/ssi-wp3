from typing import Tuple
from wordcloud import WordCloud
import pandas as pd
import os


def series_to_set(series: pd.Series) -> set:
    """Converts a pandas series to a set"""
    return set(series.drop_duplicates().str.replace('[^0-9a-zA-Z.,-/ ]', "", regex=True).str.lstrip().str.rstrip().str.lower().tolist())


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


def write_set_texts_to_file(set1, filename: str, delimiter=";", chunk_size: int = 80):
    """ Writes a set of texts to a file """
    with open(filename, "w") as text_file:
        sorted_set = sorted(set1)
        for i in range(0, len(set1), chunk_size):
            text_file.write(f'{delimiter}'.join(sorted_set[i:i+chunk_size]))
            text_file.write("\n")


def detect_product_differences(receipt_texts_before: set, receipt_texts_after: set) -> Tuple[set, set, set, set]:
    """Detects differences between two sets of texts

    Parameters
    ----------
    receipt_texts_before : set
        Set of receipt texts before, this is for example the set of receipt texts in a previous
        year or month

    receipt_texts_after : set
        Set of receipt texts after, this is for example the set of receipt texts in a current
        year or month

    Returns
    -------
    A tuple of four sets:
    - texts_kept_the_same: texts that are present in both sets
    - combined_texts: texts that are present in either set
    - texts_disappeared: texts that are present in the first set but not in the second set
    - new_texts: texts that are present in the second set but not in the first set
    """
    texts_kept_the_same = receipt_texts_before.intersection(
        receipt_texts_after)
    combined_texts = receipt_texts_before.union(receipt_texts_after)
    texts_disappeared = receipt_texts_before.difference(
        receipt_texts_after)
    new_texts = receipt_texts_after.difference(
        receipt_texts_before)

    return texts_kept_the_same, combined_texts, texts_disappeared, new_texts


def compare_receipt_texts_per_period(dataframe: pd.DataFrame, period_column: str, receipt_text_column: str) -> pd.DataFrame:
    """Compares receipt texts per period"""
    receipt_texts_per_period = dataframe.groupby(
        period_column)[receipt_text_column].apply(series_to_set)

    for period in receipt_texts_per_period.index:
        # Detect which products disappeared and which products are new
        # Add this as a column to the dataframe
        period_texts = receipt_texts_per_period[period]
        new_texts_column = [False for _ in period_texts]

        other_periods = receipt_texts_per_period[other_period.index != period]
        for other_period in other_periods.index:
            _, _, _, new_texts = detect_product_differences(
                period_texts, other_periods[other_period])
            new_texts_column.extend(
                [True if text in new_texts else False for text in other_periods[other_period]])

        dataframe[f"new_text_{period}"] = new_texts_column

    return dataframe


def compare_receipt_texts(receipt_texts_left: set, receipt_texts_right: set, output_directory: str, supermarket_name: str, name_left: str = "left", name_right: str = "right"):
    """Compares two receipt texts"""
    texts_kept_the_same, combined_texts, text_disappeared, new_texts = detect_product_differences(
        receipt_texts_left, receipt_texts_right)
    write_set_texts_to_file(texts_kept_the_same, os.path.join(
        output_directory, f"{supermarket_name}_{name_left}_{name_right}_kept_the_same.txt"))
    write_set_texts_to_file(combined_texts, os.path.join(
        output_directory, f"{supermarket_name}_{name_left}_{name_right}_all_texts.txt"))
    write_set_texts_to_file(text_disappeared, os.path.join(
        output_directory, f"{supermarket_name}_{name_left}_{name_right}_texts_disappeared.txt"))
    write_set_texts_to_file(new_texts, os.path.join(
        output_directory, f"{supermarket_name}_{name_left}_{name_right}_new_texts.txt"))
    return {
        "name_left": name_left,
        "name_right": name_right,
        "jaccard_index": jaccard_index(receipt_texts_left, receipt_texts_right),
        "dice_coefficient": dice_coefficient(receipt_texts_left, receipt_texts_right),
        "overlap_coefficient": overlap_coefficient(receipt_texts_left, receipt_texts_right),
        "left_set_length": len(receipt_texts_left),
        "right_set_length": len(receipt_texts_right),
        "intersection_length": len(texts_kept_the_same),
        "union_length": len(combined_texts),
        "left_difference_length": len(text_disappeared),
        "right_difference_length": len(new_texts),
    }


def compare_groups(receipt_text_groups, output_directory: str, supermarket_name: str,) -> pd.DataFrame:
    return pd.DataFrame([compare_receipt_texts(receipt_texts_left, receipt_texts_right, output_directory, supermarket_name, name_left, name_right)
                         for name_left, name_right, receipt_texts_left, receipt_texts_right in zip(receipt_text_groups.index, receipt_text_groups[1:].index, receipt_text_groups, receipt_text_groups[1:])
                         ])


def compare_receipt_texts_per_year(dataframe: pd.DataFrame,
                                   output_directory: str,
                                   supermarket_name: str,
                                   year_column: str = "year",
                                   receipt_text_column: str = "receipt_text") -> pd.DataFrame:
    """Compares receipt texts per year"""
    receipt_texts_per_year = dataframe.groupby(
        year_column)[receipt_text_column].apply(series_to_set)
    return compare_groups(receipt_texts_per_year, output_directory, supermarket_name)


def compare_receipt_texts_per_month(dataframe: pd.DataFrame,
                                    output_directory: str,
                                    supermarket_name: str,
                                    month_column: str = "month",
                                    receipt_text_column: str = "receipt_text") -> pd.DataFrame:
    """Compares receipt texts per month"""
    receipt_texts_per_month = dataframe.groupby(
        month_column)[receipt_text_column].apply(series_to_set)
    return compare_groups(receipt_texts_per_month, output_directory, supermarket_name)


def analyze_supermarket_receipts(filename: str,
                                 supermarket_name: str,
                                 output_directory: str,
                                 year_column: str = "year",
                                 month_column: str = "year_month",
                                 receipt_text_column: str = "receipt_text"):
    supermarket_dataframe = pd.read_parquet(filename, engine="pyarrow")
    receipts_per_year = compare_receipt_texts_per_year(
        supermarket_dataframe, output_directory, supermarket_name, year_column, receipt_text_column)
    receipts_per_year.to_csv(
        os.path.join(output_directory, f"{supermarket_name}_receipts_per_year.csv"))

    receipts_per_year.plot.bar(subplots=True)[0].figure.savefig(os.path.join(
        output_directory, f"{supermarket_name}_receipts_per_year.png"))

    receipts_per_month = compare_receipt_texts_per_month(
        supermarket_dataframe, output_directory, supermarket_name, month_column, receipt_text_column)
    receipts_per_month.to_csv(
        os.path.join(output_directory, f"{supermarket_name}_receipts_per_month.csv"))
    receipts_per_month.plot.bar(subplots=True)[0].figure.savefig(os.path.join(
        output_directory, f"{supermarket_name}_receipts_per_month.png"))
