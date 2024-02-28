from text_analysis import series_to_set
import pandas as pd
import numpy as np


def unique_texts_and_eans(dataframe: pd.DataFrame,
                          receipt_text_column: str = "receipt_text",
                          product_id_column: str = "ean_number"
                          ) -> pd.DataFrame:
    """ This function calculates the number of unique texts and EANs for the
    complete file.

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

    pd.DataFrame
        A dataframe containing the number of unique texts and EANs.
    """
    return dataframe[[receipt_text_column, product_id_column]].nunique()


def unique_texts_and_eans_per_period(dataframe: pd.DataFrame,
                                     period_column: str = "year_month",
                                     receipt_text_column: str = "receipt_text",
                                     product_id_column: str = "ean_number"
                                     ) -> pd.DataFrame:
    """ This function calculates the number of unique texts and EANs per period.

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

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique texts and EANs per period.
    """

    return dataframe.groupby(by=[period_column])[[receipt_text_column, product_id_column]].nunique()


def unique_texts_and_eans_per_coicop(dataframe: pd.DataFrame,
                                     coicop_column: str = "coicop_level_1",
                                     receipt_text_column: str = "receipt_text",
                                     product_id_column: str = "ean_number"
                                     ) -> pd.DataFrame:
    """ This function calculates the number of unique texts and EANs per COICOP.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    coicop_column : str
        The column containing the COICOP information. By default, it is "coicop_level_1".
        Pass a different column name to group by other (deeper) COICOP levels.

    receipt_text_column : str
        The column containing the receipt text. By default, it is "receipt_text".

    product_id_column : str
        The column containing the product ID. By default, it is "ean_number".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique texts and EANs per COICOP.    
    """
    return dataframe.groupby(
        by=[coicop_column])[[receipt_text_column, product_id_column]].nunique()


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


def texts_per_period(dataframe: pd.DataFrame,
                     period_column: str = "year_month",
                     receipt_text_column: str = "receipt_text",
                     product_id_column: str = "ean_number"
                     ) -> pd.DataFrame:
    grouped_texts_per_month = dataframe.groupby(
        by=period_column)[receipt_text_column].apply(series_to_set)
    grouped_texts_per_month = grouped_texts_per_month.reset_index()

    # TODO: finish!
    return grouped_texts_per_month
