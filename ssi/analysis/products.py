import pandas as pd


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

    Returns
    -------
    pd.DataFrame
        A dataframe containing the number of unique texts and EANs per COICOP.    
    """
    return dataframe.groupby(
        by=[coicop_column])[[receipt_text_column, product_id_column]].nunique()
