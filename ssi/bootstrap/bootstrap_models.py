from typing import Any, Dict
from .bootstrap import BootstrapSample, perform_bootstrap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
from functools import partial
import nlpaug.augmenter.char as nac
import pandas as pd
import tqdm


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


def evaluate_model(y_pred, y_true) -> Dict[str, Any]:
    eval_dict = dict()
    metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        "precision_micro": partial(precision_score, average="micro"),
        "precision_macro": partial(precision_score, average="macro"),
        "precision_weighted": partial(precision_score, average="weighted"),
        "recall_micro": partial(recall_score, average="micro"),
        "recall_macro": partial(recall_score, average="macro"),
        "recall_weighted": partial(recall_score, average="weighted"),
        "f1_micro": partial(f1_score, average="micro"),
        "f1_macro": partial(f1_score, average="macro"),
        "f1_weighted": partial(f1_score, average="weighted"),
        # 'roc_auc_macro_ovr': partial(roc_auc_score, average='macro', multi_class='ovr'),
        # 'roc_auc_weighted_ovr': partial(roc_auc_score, average='weighted', multi_class='ovr'),
        # 'roc_auc_marco_ovo': partial(roc_auc_score, average='macro', multi_class='ovo'),
        # 'roc_auc_weighted_ovo': partial(roc_auc_score, average='weighted', multi_class='ovo'),
        'confusion_matrix': confusion_matrix
    }

    for metric_name, metric_function in metrics.items():
        eval_dict[metric_name] = metric_function(y_true, y_pred)

    classification_report_dict = classification_report(
        y_true, y_pred, output_dict=True)

    for class_name, class_metrics in classification_report_dict.items():
        if class_name == 'accuracy':
            eval_dict['accuracy'] = class_metrics
            continue

        for metric_name, metric_value in class_metrics.items():
            eval_dict[f'{class_name}_{metric_name}'] = metric_value
    return eval_dict


def sklearn_evaluation_function(bootstrap_index: int,
                                total_number_bootstraps: int,
                                sample: BootstrapSample,
                                sklearn_pipeline,
                                parameter_sampler,
                                feature_column: str,
                                label_column: str,
                                progress_bar: tqdm.tqdm) -> Dict[str, Any]:

    # Train the model
    train_sample_df = sample.bootstrap_sample

    # Get new parameter combinations from parameter sampler
    params = next(parameter_sampler)
    sklearn_pipeline.set_params(**params)

    sklearn_pipeline.fit(
        train_sample_df[feature_column], train_sample_df[label_column])

    # Evaluate the model on the out-of-bag sample
    test_sample_df = sample.oob_sample
    y_pred = sklearn_pipeline.predict(test_sample_df[feature_column])
    y_true = test_sample_df[label_column].values

    eval_dict = evaluate_model(y_pred, y_true)
    eval_dict['bootstrap_index'] = bootstrap_index

    progress_bar.update(1)
    return eval_dict


def bootstrap_model(sklearn_pipeline,
                    parameter_sampler,
                    dataframe: pd.DataFrame,
                    n_bootstraps: int,
                    n_samples_per_bootstrap: int,
                    feature_column: str,
                    label_column: str,
                    random_state: int = 42) -> pd.DataFrame:
    with tqdm.tqdm(total=n_bootstraps) as progress_bar:
        return perform_bootstrap(dataframe=dataframe,
                                 n_bootstraps=n_bootstraps,
                                 n_samples_per_bootstrap=n_samples_per_bootstrap,
                                 evaluation_function=sklearn_evaluation_function,
                                 replace=True,
                                 random_state=random_state,
                                 sklearn_pipeline=sklearn_pipeline,
                                 parameter_sampler=parameter_sampler,
                                 feature_column=feature_column,
                                 label_column=label_column,
                                 progress_bar=progress_bar
                                 )
