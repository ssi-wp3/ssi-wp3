from typing import Any, Dict
from .bootstrap import BootstrapSample, perform_bootstrap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
from ..machine_learning.evaluation.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score
from functools import partial
import nlpaug.augmenter.char as nac
import pandas as pd
import tqdm
import numpy as np


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


def evaluate_model(y_true: np.array, y_pred: np.array) -> Dict[str, Any]:
    eval_dict = dict()
    metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'confusion_matrix': confusion_matrix,
        'hierarchical_precision_binary': hierarchical_precision_score,
        'hierarchical_recall_binary': hierarchical_recall_score,
        'hierarchical_f1_binary': hierarchical_f1_score
    }

    metric_average_functions = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "hierarchical_precision": hierarchical_precision_score,
        "hierarchical_recall": hierarchical_recall_score,
        "hierarchical_f1": hierarchical_f1_score
    }

    for metric_name, metric_function in metric_average_functions.items():
        for metric_average in ['micro', 'macro', 'weighted']:
            metrics[f"{metric_name}_{metric_average}"] = partial(
                metric_function, average=metric_average)

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
    print(f"Evaluating model with parameters: \n\n{params}")

    sklearn_pipeline.set_params(**params)

    sklearn_pipeline.fit(
        train_sample_df[feature_column], train_sample_df[label_column])

    # Evaluate the model on the out-of-bag sample
    test_sample_df = sample.oob_sample
    y_pred = sklearn_pipeline.predict(test_sample_df[feature_column])
    y_true = test_sample_df[label_column].values

    eval_dict = evaluate_model(y_true, y_pred)

    eval_params_dict = dict()
    eval_params_dict['bootstrap_index'] = bootstrap_index

    for key, value in params.items():
        eval_params_dict[key] = value

    for key, value in eval_dict.items():
        eval_params_dict[key] = value

    progress_bar.update(1)
    return eval_params_dict


def bootstrap_model(sklearn_pipeline,
                    parameter_sampler,
                    dataframe: pd.DataFrame,
                    results_file,
                    n_bootstraps: int,
                    n_samples_per_bootstrap: int,
                    feature_column: str,
                    label_column: str,
                    random_state: int = 42) -> pd.DataFrame:
    with tqdm.tqdm(total=n_bootstraps) as progress_bar:
        perform_bootstrap(dataframe=dataframe,
                          n_bootstraps=n_bootstraps,
                          n_samples_per_bootstrap=n_samples_per_bootstrap,
                          results_file=results_file,
                          evaluation_function=sklearn_evaluation_function,
                          replace=True,
                          random_state=random_state,
                          sklearn_pipeline=sklearn_pipeline,
                          parameter_sampler=parameter_sampler,
                          feature_column=feature_column,
                          label_column=label_column,
                          progress_bar=progress_bar
                          )
