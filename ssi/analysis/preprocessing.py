import pandas as pd
from typing import List, Dict, Callable
from transformers import AutoTokenizer
from ..constants import Constants


def split_strings(string_column: pd.Series, separator: str = ' ') -> pd.Series:
    """ Split strings in a column into separate words.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to split.

    separator : str
        The separator to use to split the strings. By default, the separator is a space.

    Returns
    -------
    pd.Series
        A series with the unique split strings.
    """
    return string_column.str.split(separator).explode()


def tokenize_strings(string_column: pd.Series, tokenizer: Callable[[str], List[str]]) -> pd.Series:
    """ Tokenize strings in a column using a custom tokenizer.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to tokenize.

    tokenizer : Callable[[str], List[str]]
        A function that takes a string as input and returns a list of tokens.

    Returns
    -------
    pd.Series
        A series with the unique tokens.
    """
    return string_column.apply(tokenizer).explode()


def drop_short_strings(string_column: pd.Series, drop_less_than: int = 3):
    """ 
    Remove all entries with strings shorter than "drop_less_than".

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to tokenize.

    Returns
    -------
    pd.Series
        A series with the unique tokens.
    """
    if drop_less_than < 1:
        print("WARNING: Minimum string length less than 1.")

    return string_column[string_column.str.len() >= drop_less_than]


def huggingface_tokenize_strings(string_column: pd.Series, tokenizer_name: str = "gpt2") -> pd.Series:
    """ Tokenize strings in a column using a Hugging Face tokenizer.

    Parameters
    ----------
    string_column : pd.Series
        The column containing the strings to tokenize.

    tokenizer_name : str
        The name of the Hugging Face tokenizer to use. By default, the GPT-2 tokenizer is used.

    Returns
    -------
    pd.Series
        A series with the unique tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenize_strings(string_column, tokenizer.tokenize)


class Preprocessing:
    @property
    def all_preprocessing_functions(self) -> Dict[str, Dict[str, Callable[[pd.Series], pd.Series]]]:
        return {
            Constants.PRODUCT_ID_COLUMN: {
                "raw": lambda x: x
            },
            Constants.RECEIPT_TEXT_COLUMN: {
                "raw": lambda x: x,
                "lower": lambda x: x.str.lower(),
                "lower_split_words": lambda x: split_strings(x.str.lower()),
                "stripped": lambda x: x.str.replace('[^0-9a-zA-Z.,-/ ]', '', regex=True).str.lstrip().str.lower(),
                "stripped_split_words": lambda x: split_strings(x.str.replace('[^0-9a-zA-Z.,-/ ]', '', regex=True).str.strip().str.lower()),
                "stripped_split_alpha_drop_short": lambda x: drop_short_strings(split_strings(x.str.replace('[^a-z\s]', '', regex=True).str.strip().str.lower()), drop_less_than=3),
                "raw_tokenized_gpt2": lambda x: huggingface_tokenize_strings(x, "gpt2"),
                "raw_tokenized_mini_lm": lambda x: huggingface_tokenize_strings(x, "sentence-transformers/all-MiniLM-L6-v2"),
                "raw_tokenized_labse": lambda x: huggingface_tokenize_strings(x, "sentence-transformers/LaBSE"),
            }
        }

    @property
    def ean_number_preprocessing_functions(self) -> Dict[str, Callable[[pd.Series], pd.Series]]:
        return self.all_preprocessing_functions[Constants.PRODUCT_ID_COLUMN]

    @property
    def receipt_text_preprocessing_functions(self) -> Dict[str, Callable[[pd.Series], pd.Series]]:
        return self.all_preprocessing_functions[Constants.RECEIPT_TEXT_COLUMN]
