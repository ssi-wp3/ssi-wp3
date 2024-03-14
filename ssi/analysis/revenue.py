from typing import List
import pandas as pd


def total_revenue(dataframe: pd.DataFrame,
                  amount_column: str,
                  revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for all products in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------

    pd.DataFrame
        A dataframe containing the total amount and total revenue.    
    """
    total_amount = dataframe[amount_column].sum()
    total_revenue = dataframe[revenue_column].sum()
    return pd.DataFrame({'Total Amount': [total_amount], 'Total Revenue': [total_revenue]})


def total_revenue_per_products(dataframe: pd.DataFrame,
                               product_id_column: str,
                               amount_column: str,
                               revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each product in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    product_id_column : str
        The name of the column containing the product id.   

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the product id, total amount and total revenue.    
    """
    return dataframe.groupby(product_id_column).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()


def total_revenue_per_coicop(dataframe: pd.DataFrame,
                             coicop_column: str,
                             amount_column: str,
                             revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each coicop in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    coicop_column : str
        The name of the column containing the coicop code.   

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the coicop code, total amount and total revenue.    
    """
    return dataframe.groupby(coicop_column).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()


def total_revenue_per_period(dataframe: pd.DataFrame,
                             period_column: str,
                             amount_column: str,
                             revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each period in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    period_column : str
        The name of the column containing the period.   

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the period, total amount and total revenue.    
    """
    return dataframe.groupby(period_column).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()


def total_revenue_per_coicop_and_period(dataframe: pd.DataFrame,
                                        period_column: str,
                                        coicop_column: str,
                                        amount_column: str,
                                        revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each product and period in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    product_id_column : str
        The name of the column containing the product id.   

    period_column : str
        The name of the column containing the period.   

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the product id, period, total amount and total revenue.    
    """
    return dataframe.groupby([period_column, coicop_column]).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()


def revenue_for_coicop_hierarchy(dataframe: pd.DataFrame,
                                 coicop_columns: List[str],
                                 amount_column: str,
                                 revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each coicop subsequent coicop level in the dataframe.
    Return each level of the hierarchy as a column in the dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    coicop_columns : List[str]
        The list of column names containing the subsequent coicop levels.

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the total amount and total revenue for each subsequent coicop level.
        For each coicop level analysed there's a column in the dataframe containing the coicop label.
        The amount and revenue columns specify the values for each leaf in the coicop hierarchy.
        This dataframe can be used to create sunburst plot of the product revenue.          
    """
    return dataframe.groupby(coicop_columns).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()


def product_revenue_versus_lifetime(dataframe: pd.DataFrame,
                                    lifetime_column: str,
                                    product_id_column: str,
                                    amount_column: str,
                                    revenue_column: str) -> pd.DataFrame:
    """ Calculate the total revenue for each lifetime in the dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the revenue data.

    lifetime_column : str
        The name of the column containing the lifetime.   

    amount_column : str
        The name of the column containing the amount of products.

    revenue_column : str
        The name of the column containing the revenue.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the lifetime, total amount and total revenue.    
    """
    pass
    # return dataframe.groupby(lifetime_column).apply(lambda x: total_revenue(x, amount_column, revenue_column)).reset_index()
