from typing import Dict, Any, Optional
from collections import OrderedDict
from ..constants import Constants
import argparse
import pandas as pd
import os
import tqdm
import numpy as np


def get_column_types(filename: str) -> Optional[OrderedDict[str, Any]]:
    # From Justin's code, but converted them to english and lower case
    column_types = OrderedDict([
        ('bg_number', str),
        ('month', str),
        (Constants.COICOP_LABEL_COLUMN, str),
        ('coicop_name', str),
        ('isba_number', str),
        ('isba_name', str),
        ('esba_number', str),
        ('esba_name', str),
        ('rep_id', str),
        (Constants.PRODUCT_ID_COLUMN, str),
        ('ean_name', str),
        ('revenue', np.float32),
        ('amount', np.float32)
    ])

    if filename.lower().startswith("omzeteans"):
        return column_types
    elif filename.lower().startswith("output"):
        return OrderedDict([(column_name, column_types[column_name])
                            for column_name in ['bg_number', Constants.COICOP_LABEL_COLUMN, 'coicop_name', 'isba_number', 'isba_name', 'esba_number', 'esba_name', 'rep_id', Constants.PRODUCT_ID_COLUMN, 'ean_name']
                            ])
    return None


def get_columns_to_rename(filename: str) -> Optional[Dict[str, str]]:
    basename = os.path.basename(filename)
    if basename.lower().startswith("kassabon") or basename.lower().startswith("receipts"):
        return {
            'Datum_vanaf': Constants.START_DATE_COLUMN,
            'Ean': Constants.PRODUCT_ID_COLUMN,
            'Kassabon': Constants.RECEIPT_TEXT_COLUMN,
            'RPK_REP_id': 'rep_id'
        }
    return None


def check_header(input_file, delimiter: str) -> bool:
    first_line = pd.read_csv(input_file, sep=delimiter, nrows=1)
    standard_header = {"bgnr", "maand", "coicopnr", "coicopnaam", "isbanr",
                       "isbanaam", "esbanr", "esbanaam", "repid", "eannr",
                       "eannaam", "omzet", "aantal"}
    column_names = set([column.lower()
                       for column in first_line.columns.tolist()])
    return len(column_names.intersection(standard_header)) > 0


def convert_to_parquet(input_filename: str,
                       input_file,
                       output_file,
                       delimiter: str = ";",
                       encoding: str = "utf-8",
                       extension: str = ".csv",
                       decimal: str = ",") -> None:

    filename = os.path.basename(input_filename).replace(extension, "")

    has_header = check_header(input_file,
                              delimiter)
    # Add header names and types to all but kassa files
    header_types = get_column_types(filename)
    header_names = None if not header_types else [
        name for name in header_types.keys()]

    skipheader = 1 if has_header else 0

    df = pd.read_csv(input_filename,
                     sep=delimiter,
                     engine="pyarrow",
                     encoding=encoding,
                     decimal=decimal,
                     names=header_names,
                     dtype=header_types,
                     skiprows=skipheader,
                     parse_dates=True)

    columns_to_rename = get_columns_to_rename(filename)
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
    df.to_parquet(output_file, engine="pyarrow")
