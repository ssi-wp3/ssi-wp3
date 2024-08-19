from .bootstrap import BootstrapSample, perform_bootstrap
import nlpaug.augmenter.char as nac
import pandas as pd


def ocr_preprocessing(bootstrap_sample: BootstrapSample, receipt_text_column: str, number_of_texts: int) -> BootstrapSample:
    """ Perform OCR preprocessing augmentation.

    """

    def ocr_augment(dataframe: pd.DataFrame, receipt_text_column: str, number_of_texts: int) -> pd.DataFrame:
        aug = nac.OcrAug()
        return aug.augment(dataframe[receipt_text_column], n=number_of_texts)

    augmented_bootstrap_sample = ocr_augment(
        bootstrap_sample.training_sample, receipt_text_column, number_of_texts)
    augmented_oob_sample = ocr_augment(
        bootstrap_sample.oob_sample, receipt_text_column, number_of_texts)
    return BootstrapSample(augmented_bootstrap_sample, augmented_oob_sample)
