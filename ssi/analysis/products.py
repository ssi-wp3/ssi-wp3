from typing import Optional, List
from text_analysis import series_to_set
import pandas as pd
import numpy as np


def unique_column_values(dataframe: pd.DataFrame,
                         value_columns: List[str] = [
                             "receipt_text", "ean_number"]
                         ) -> pd.DataFrame:
    """ This function calculates the number of unique column values for the selected columns
    over the complete dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The input dataframe.

    value_columns : List[str]
        The columns to calculate the unique counts for. By default, it is ["receipt_text", "ean_number"].
        -   "receipt_text" is the column containing the receipt text. 
        -   "ean_number" is the column containing the product ID.

    Returns
    -------

    pd.DataFrame
        A dataframe containing the number of unique column values for the selected columns.
    """
    return dataframe[value_columns].nunique()


def unique_columns_values_per_coicop(dataframe: pd.DataFrame,
                                     coicop_column: str = "coicop_level_1",
                                     value_columns: List[str] = [
                                         "receipt_text", "ean_number"]
                                     ) -> pd.DataFrame:
    """ This function calculates the number of unique column values for selected columns per COICOP.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    coicop_column : str
        The column containing the COICOP information. By default, it is "coicop_level_1".
        Pass a different column name to group by other (deeper) COICOP levels.

    value_columns : List[str]
        The columns to calculate the unique counts for. By default, it is ["receipt_text", "ean_number"].
        -   "receipt_text" is the column containing the receipt text.
        -   "ean_number" is the column containing the product ID.    

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique values for the selected columns.    
    """
    return dataframe.groupby(
        by=[coicop_column])[value_columns].nunique()


def unique_column_values_per_period(dataframe: pd.DataFrame,
                                    period_column: str = "year_month",
                                    value_columns: List[str] = [
                                        "receipt_text",  "ean_number"]
                                    ) -> pd.DataFrame:
    """ This function calculates the number of unique column values per period.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    period_column : str
        The column containing the period information. By default, it is "year_month".
        Pass a different column name for other periods, for example "year". It's also
        possible to pass a list of columns to period_column to group by multiple columns.

    value_columns : List[str]
        The columns to calculate the unique counts for. By default, it is ["receipt_text", "ean_number"].
        -  "receipt_text" is the column containing the receipt text.
        -  "ean_number" is the column containing the product ID.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique column values per period.
    """

    return dataframe.groupby(by=[period_column])[value_columns].nunique()


def texts_per_ean_histogram(dataframe: pd.DataFrame,
                            receipt_text_column: str = "receipt_text",
                            product_id_column: str = "ean_number"
                            ) -> pd.Series:
    """ This function calculates the histogram of the receipt texts per EAN. 
    First the number of unique texts per EAN is calculated. Then these counts 
    are binned a histogram of the counts is created. The histogram shows how
    often a certain number of texts per EAN occurs.

    This can be used to create a histogram of the number of texts per EAN.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------

    pd.Series
        A series containing the histogram of receipt text counts per of EAN.
    """
    texts_per_ean = dataframe.groupby(by=product_id_column)[
        receipt_text_column].nunique()
    texts_per_ean = texts_per_ean.reset_index()
    receipt_text_counts = texts_per_ean.receipt_text.value_counts()
    receipt_text_counts = receipt_text_counts.sort_index()
    return receipt_text_counts


def log_texts_per_ean_histogram(dataframe: pd.DataFrame,
                                receipt_text_column: str = "receipt_text",
                                product_id_column: str = "ean_number"
                                ) -> pd.Series:
    """ This function calculates the histogram of the receipt texts per EAN,
    but instead of the function above returns the logarithm of the counts.

    First the number of unique texts per EAN is calculated. Then these counts 
    are binned a histogram of the counts is created. Then the logarithm is taken of
    these counts. The histogram shows how often a certain number of texts per EAN occurs.

    This can be used to create a histogram of the number of texts per EAN.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------

    pd.Series
        A series containing the histogram of receipt text counts per of EAN.
        The logarithm of the counts is returned instead of the count itself.
    """
    texts_per_ean_histogram = texts_per_ean_histogram(
        dataframe, receipt_text_column, product_id_column)
    return np.log(texts_per_ean_histogram)


def add_lagged_columns(grouped_texts_eans_per_month: pd.DataFrame,
                       receipt_text_column: str,
                       product_id_column: str) -> pd.DataFrame:
    """ This function adds lagged columns to the dataframe. The lagged columns contain the values
    of the previous period for the receipt texts and the EANs.

    Parameters
    ----------
    grouped_texts_eans_per_month : pd.DataFrame
        The input dataframe.

    receipt_text_column : str
        The column containing the receipt text.

    product_id_column : str
        The column containing the product ID.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the lagged columns for the receipt texts and the EANs.
    """
    grouped_texts_eans_per_month[f"{receipt_text_column}_lagged"] = grouped_texts_eans_per_month[receipt_text_column].shift(
        1)
    grouped_texts_eans_per_month[f"{product_id_column}_lagged"] = grouped_texts_eans_per_month[product_id_column].shift(
        1)
    return grouped_texts_eans_per_month


def products_per_period(dataframe: pd.DataFrame,
                        period_column: str = "year_month",
                        receipt_text_column: str = "receipt_text",
                        product_id_column: str = "ean_number"
                        ) -> pd.DataFrame:
    """ This function creates a dataframe that contains the unique receipt texts and
    product identifiers per period in "period column". The dataframe contains a column
    with a set of unique receipt texts and a column with a set of unique product identifiers.
    In addition, the dataframe contains a lagged version, i.e. a column containing the values
    of the previous period of these columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    period_column : str
        The column containing the period information. By default, it is "year_month".
        Pass a different column name for other periods, for example "year". It's also
        possible to pass a list of columns to period_column to group by multiple columns.

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------

    pd.DataFrame
        A dataframe containing the unique receipt texts and product identifiers per period.
        The dataframe also contains a lagged version of these columns.
    """
    grouped_texts_per_month = dataframe.groupby(
        by=period_column)[receipt_text_column].apply(series_to_set)
    grouped_texts_per_month = grouped_texts_per_month.reset_index()

    grouped_eans_per_month = dataframe.groupby(
        by=period_column)[product_id_column].apply(series_to_set)
    grouped_eans_per_month = grouped_eans_per_month.reset_index()

    grouped_texts_eans_per_month = grouped_texts_per_month.merge(
        grouped_eans_per_month, on=period_column)

    grouped_texts_eans_per_month = add_lagged_columns(grouped_texts_eans_per_month,
                                                      receipt_text_column, product_id_column)

    return grouped_texts_eans_per_month


def intersection(left_column: Optional[set], right_column: Optional[set]) -> Optional[set]:
    """This function calculates the intersection of two sets. The sets are columns in a dataframe.
    This function serves as a helper function that can be passed to the apply method of a dataframe.

    Parameters
    ----------

    left_column : Optional[set]
        The left column containing a set of elements.

    right_column : Optional[set]
        The right column containing a set of elements.

    Returns
    -------
    Optional[set]
        The intersection of the two sets.
    """
    if not left_column or not right_column:
        return None
    return left_column.intersection(right_column)


def introduced_products(left_column: Optional[set], right_column: Optional[set]) -> Optional[set]:
    """ This function calculates the introduced products. The sets are columns in a dataframe.
    This function serves as a helper function that can be passed to the apply method of a dataframe.

    Parameters
    ----------

    left_column : Optional[set]
        The left column containing a set of elements.

    right_column : Optional[set]
        The right column containing a set of elements.

    Returns
    -------
    Optional[set]
        The introduced products.
    """
    if not left_column or not right_column:
        return None
    return left_column.difference(right_column)


def removed_products(left_column: Optional[set], right_column: Optional[set]) -> Optional[set]:
    """ This function calculates the removed products. The sets are columns in a dataframe.
    This function serves as a helper function that can be passed to the apply method of a dataframe.

    Parameters
    ----------
    left_column : Optional[set]
        The left column containing a set of elements.

    right_column : Optional[set]
        The right column containing a set of elements.

    Returns
    -------

    Optional[set]
        The removed products.
    """
    if not left_column or not right_column:
        return None
    return right_column.difference(left_column)


def number_of_products(column: Optional[set]) -> int:
    """ This function calculates the number of products in a set. The set is a column in a dataframe.
    This function serves as a helper function that can be passed to the apply method of a dataframe.

    Parameters
    ----------

    column : Optional[set]
        The column containing a set of elements.

    Returns
    -------
    int
        The number of products in the set.

    """
    if not column:
        return 0
    return len(column)


def compare_products_per_period(dataframe: pd.DataFrame,
                                period_column: str = "year_month",
                                receipt_text_column: str = "receipt_text",
                                product_id_column: str = "ean_number"
                                ) -> pd.DataFrame:
    """This functions compares the receipt texts and product identifiers per period with those of
    the last period. It returns the texts and product identifiers that are the same, introduced, or 
    removed, as well as, the number of texts and product identifiers for all of those changes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    period_column : str
        The column containing the period information. By default, it is "year_month".
        Pass a different column name for other periods, for example "year". It's also
        possible to pass a list of columns to period_column to group by multiple columns.

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the comparison of the receipt texts and product identifiers per period. 

    """
    texts_per_period_df = products_per_period(
        dataframe, period_column, receipt_text_column, product_id_column)

    for column in [receipt_text_column, product_id_column]:
        texts_per_period_df[f"{column}_same"] = texts_per_period_df.apply(
            lambda row: intersection(row[column], row[f"{column}_lagged"]), axis=1)
        texts_per_period_df[f"{column}_introduced"] = texts_per_period_df.apply(
            lambda row: introduced_products(row[column], row[f"{column}_lagged"]), axis=1)
        texts_per_period_df[f"{column}_removed"] = texts_per_period_df.apply(
            lambda row: removed_products(row[column], row[f"{column}_lagged"]), axis=1)

        texts_per_period_df[f"number_{column}_same"] = texts_per_period_df[f"{column}_same"].apply(
            number_of_products)
        texts_per_period_df[f"number_{column}_introduced"] = texts_per_period_df[f"{column}_introduced"].apply(
            number_of_products)
        texts_per_period_df[f"number_{column}_removed"] = texts_per_period_df[f"{column}_removed"].apply(
            number_of_products)


def products_per_period_coicop_level(dataframe: pd.DataFrame,
                                     period_column: str = "year_month",
                                     coicop_level_column: str = "coicop_level_1",
                                     receipt_text_column: str = "receipt_text",
                                     product_id_column: str = "ean_number"
                                     ) -> pd.DataFrame:
    """ This function creates a dataframe that contains the unique receipt texts and
    product identifiers per period and COICOP level. The dataframe contains a column
    with a set of unique receipt texts and a column with a set of unique product identifiers.
    In addition, the dataframe contains a lagged version, i.e. a column containing the values
    of the previous period for these columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    period_column : str
        The column containing the period information. By default, it is "year_month".

    coicop_level_column : str
        The column containing the COICOP level information. By default, it is "coicop_level_1".

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the unique receipt texts and product identifiers per period and COICOP level.
        The dataframe also contains a lagged version of these columns.
    """
    grouped_texts_per_month_coicop = dataframe.groupby(
        by=[period_column, coicop_level_column])[receipt_text_column].apply(series_to_set)
    grouped_texts_per_month_coicop = grouped_texts_per_month_coicop.reset_index()
    grouped_eans_per_month_coicop = dataframe.groupby(
        by=[period_column, coicop_level_column])[product_id_column].apply(series_to_set)
    grouped_eans_per_month_coicop = grouped_eans_per_month_coicop.reset_index()

    grouped_texts_eans_per_month_coicop = grouped_texts_per_month_coicop.merge(
        grouped_eans_per_month_coicop, on=[period_column, coicop_level_column])

    # TODO: this doesn't work, needs to be per group
    grouped_eans_per_month_coicop = add_lagged_columns(grouped_eans_per_month_coicop,
                                                       receipt_text_column, product_id_column)
    return grouped_texts_eans_per_month_coicop


def compare_products_per_period_coicop_level(dataframe: pd.DataFrame,
                                             period_column: str = "year_month",
                                             coicop_level_column: str = "coicop_level_1",
                                             receipt_text_column: str = "receipt_text",
                                             product_id_column: str = "ean_number"
                                             ) -> pd.DataFrame:
    texts_per_period_coicop_df = products_per_period_coicop_level(
        dataframe, period_column, coicop_level_column, receipt_text_column, product_id_column)
