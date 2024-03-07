import pandas as pd
from .train_model import train_and_evaluate_model


def create_combined_dataframe(store1_dataframe: pd.DataFrame,
                              store2_dataframe: pd.DataFrame,
                              store_id_column: str,
                              receipt_text_column: str
                              ) -> pd.DataFrame:
    """Combine two store dataframes into one dataframe for adversarial validation.
    The two dataframes are combined by concatenating the receipt text columns and the store id columns.
    The machine learning model can then be trained to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text    
    """
    return pd.concat([
        store1_dataframe[[store_id_column, receipt_text_column]],
        store2_dataframe[[store_id_column, receipt_text_column]]
    ])


def filter_unique_combinations(dataframe: pd.DataFrame,
                               store_id_column: str,
                               receipt_text_column: str
                               ) -> pd.DataFrame:
    """Filter the dataframe to only contain unique combinations of store id and receipt text.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to filter

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text    
    """
    return dataframe.drop_duplicates(subset=[store_id_column, receipt_text_column])


def train_adversarial_model(store1_dataframe: pd.DataFrame,
                            store2_dataframe: pd.DataFrame,
                            store_id_column: str,
                            receipt_text_column: str,
                            model,
                            test_size: float = 0.2,
                            number_of_jobs: int = -1,
                            verbose: bool = False
                            ):
    """Train an adversarial model to predict the store id based on the receipt text column.

    Parameters
    ----------
    store1_dataframe : pd.DataFrame
        The first store dataframe

    store2_dataframe : pd.DataFrame
        The second store dataframe

    store_id_column : str
        The column name containing the store id

    receipt_text_column : str
        The column name containing the receipt text

    model : object
        The model to use

    test_size : float, optional
        The test size for the train/test split, by default 0.2

    number_of_jobs : int, optional
        The number of jobs to use, by default -1

    verbose : bool, optional
        Verbose output, by default False

    Returns
    -------
    object
        The trained model
    """
    combined_dataframe = create_combined_dataframe(
        store1_dataframe, store2_dataframe, store_id_column, receipt_text_column)
    unique_dataframe = filter_unique_combinations(
        combined_dataframe, store_id_column, receipt_text_column)
    pipeline, evaluation = train_and_evaluate_model(unique_dataframe, receipt_text_column,
                                                    store_id_column, model, test_size, number_of_jobs, verbose)
    return pipeline, evaluation
