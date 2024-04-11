from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np


class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate_training(self, filename: str, y_true_column: str, y_pred_column: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, filename: str, y_true_column: str, y_pred_column: str, **kwargs) -> Dict[str, Any]:
        pass


class ConfusionMatrixEvaluator(ModelEvaluator):
    def evaluate_training(self, filename: str,  y_true_column: str, y_pred_column: str, **kwargs) -> Dict[str, Any]:
        return self.evaluate(filename, y_true_column, y_pred_column)

    def evaluate(self, filename: str, y_true_column: str, y_pred_column: str, **kwargs) -> Dict[str, Any]:
        dataframe = pd.read_parquet(
            filename, engine="pyarrow", columns=[y_true_column, y_pred_column])

        y_true = dataframe[y_true_column]
        y_pred = dataframe[y_pred_column]

        evaluation_dict = {"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                           "accuracy": accuracy_score(y_true, y_pred),
                           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                           "classification_report": classification_report(y_true, y_pred, output_dict=True),
                           }
        for average_method in ['micro', 'macro', 'weighted']:
            evaluation_dict[f"precision_{average_method}"] = precision_score(
                y_true, y_pred, average=average_method)
            evaluation_dict[f"recall_{average_method}"] = recall_score(
                y_true, y_pred, average=average_method)
            evaluation_dict[f"f1_score_{average_method}"] = f1_score(
                y_true, y_pred, average=average_method)

        return evaluation_dict

# TODO: there is a bug in the code, the confusion matrix is not calculated correctly!!!


class ConfusionMatrix:
    def __init__(self, confusion_matrix: np.array, label_mapping: Dict[str, int]):
        # TODO: we have to seem N+1 classes, why?
        self.__label_mapping = label_mapping
        self.__true_positives, self.__false_positives, self.__true_negatives, self.__false_negatives = self._calculate_confusion_matrix_statistics(
            confusion_matrix)

    @property
    def label_mapping(self) -> Dict[str, int]:
        return self.__label_mapping

    @property
    def true_positive(self) -> np.array:
        return self.__true_positives

    @property
    def false_positive(self) -> np.array:
        return self.__false_positives

    @property
    def true_negative(self) -> np.array:
        return self.__true_negatives

    @property
    def false_negative(self) -> np.array:
        return self.__false_negatives

    @property
    def precision_score(self) -> Dict[str, float]:
        """ Calculate the precision score for each class. The precision is calculated
        as the number of true positives divided by the sum of true positives and false positives:

        precision = TP / (TP + FP)

        Returns:
        --------
        np.array
            An array of precision scores for each class.
        """
        precision_values = self.true_positive / \
            (self.true_positive + self.false_positive)
        return self.__add_labels(precision_values)

    @property
    def recall_score(self) -> Dict[str, float]:
        """ Calculate the recall score for each class. The recall
        is calculated as the number of true positives divided by the sum of true positives and false negatives:

        recall = TP / (TP + FN)

        Returns:
        --------
        np.array
            An array of recall scores for each class.
        """
        recall_values = self.true_positive / \
            (self.true_positive + self.false_negative)
        return self.__add_labels(recall_values)

    @property
    def f1_score(self) -> Dict[str, float]:
        """ Calculate the F1 score for each class. The F1 score is the harmonic mean of the precision and recall:

        F1 = 2 * (precision * recall) / (precision + recall)

        Returns:
        --------
        np.array
            An array of F1 scores for each class.
        """
        precision_scores = np.array(list(self.precision_score.values()))
        recall_scores = np.array(list(self.recall_score.values()))

        f1_values = 2 * (precision_scores * recall_scores) / \
            (precision_scores + recall_scores)
        return self.__add_labels(f1_values)

    @property
    def accuracy_score(self) -> Dict[str, float]:
        """ Calculate the accuracy score for each class. The accuracy score is calculated as the sum of true positives and true negatives divided by the sum of true positives, true negatives, false positives, and false negatives:

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        Returns:
        --------
        np.array
            An array of accuracy scores for each class.
        """
        accuracy_values = (self.true_positive + self.true_negative) / (
            self.true_positive + self.true_negative + self.false_positive + self.false_negative)
        return self.__add_labels(accuracy_values)

    def _calculate_confusion_matrix_statistics(self, confusion_matrix: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
        """ Takes a multi-class confusion matrix and returns the statistics for each class.

        Parameters:
        -----------

        confusion_matrix: np.array
            A multi-class confusion matrix

        Returns:
        --------
        ConfusionMatrix
            A named tuple containing the statistics (TP, FP, TN, FN) for each class

        For more info see: https://stackoverflow.com/questions/48100173/how-to-get-precision-recall-and-f-measure-from-confusion-matrix-in-python
        """
        TP = np.nan_to_num(np.diag(confusion_matrix))
        FP = np.nan_to_num(np.sum(confusion_matrix, axis=0) - TP)
        FN = np.nan_to_num(np.sum(confusion_matrix, axis=1) - TP)

        # True negatives
        num_classes = confusion_matrix.shape[0]

        TN = []
        for i in range(num_classes):
            temp = np.delete(confusion_matrix, i, axis=0)    # delete ith row
            temp = np.delete(temp, i, axis=1)
            TN.append(sum(sum(temp)))
        TN = np.nan_to_num(np.array(TN))

        return (TP, FP, TN, FN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positive":  self.__add_labels(self.true_positive.tolist()),
            "false_positive": self.__add_labels(self.false_positive.tolist()),
            "true_negative": self.__add_labels(self.true_negative.tolist()),
            "false_negative": self.__add_labels(self.false_negative.tolist()),
        }

    def __add_labels(self, metric_values):
        return {label: value for label, value in zip(self.label_mapping.keys(), metric_values)}


def get_labels_and_predictions(dataframe: pd.DataFrame, label_column: str, column_prefix: str = "predict_") -> pd.DataFrame:
    y_true = dataframe[label_column]
    y_pred = dataframe[f"{column_prefix}{label_column}"]
    return y_true, y_pred


def calculate_metrics_per_period(dataframe: pd.DataFrame, label_column: str, prediction_column: str, period_column: str) -> pd.DataFrame:
    return dataframe.groupby(period_column).apply(lambda x: pd.Series({
        "accuracy": accuracy_score(x[label_column], x[prediction_column]),
        "balanced_accuracy": balanced_accuracy_score(x[label_column], x[prediction_column]),

        "precision_micro": precision_score(x[label_column], x[prediction_column], average="micro"),
        "precision_macro": precision_score(x[label_column], x[prediction_column], average="macro"),
        "precision_weighted": precision_score(x[label_column], x[prediction_column], average="weighted"),

        "recall_micro": recall_score(x[label_column], x[prediction_column], average="micro"),
        "recall_macro": recall_score(x[label_column], x[prediction_column], average="macro"),
        "recall_weighted": recall_score(x[label_column], x[prediction_column], average="weighted"),

        "f1_micro": f1_score(x[label_column], x[prediction_column], average="micro"),
        "f1_macro": f1_score(x[label_column], x[prediction_column], average="macro"),
        "f1_weighted": f1_score(x[label_column], x[prediction_column], average="weighted"),

        "auc_roc": roc_auc_score(x[label_column], x[prediction_column])
    })).reset_index()


def calculate_metrics(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, auc_roc


def save_metrics(accuracy, precision, recall, f1, auc_roc, output_path):
    # Save metrics to disk
    with open(os.path.join(output_path, 'evaluation_metrics.txt'), 'w') as f:
        f.write(
            f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}\nAUC-ROC: {auc_roc}')


def plot_confusion_matrix(y_true, y_pred, output_path):
    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))


def plot_roc_curve(y_true, y_pred, output_path):
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_roc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'roc_curve.png'))


def plot_precision_recall_curve(y_true, y_pred, output_path):
    # Plot and save Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_score = auc(recall, precision)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision,
             label='Precision-Recall curve (area = %0.2f)' % auc_score)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'precision_recall_curve.png'))


def evaluate(dataframe: pd.DataFrame, label_column: str, output_path: str, column_prefix: str = "predict_"):
    y_true, y_pred = get_labels_and_predictions(
        dataframe, label_column, column_prefix)
    accuracy, precision, recall, f1, auc_roc = calculate_metrics(
        y_true, y_pred)
    save_metrics(accuracy, precision, recall, f1, auc_roc, output_path)
    plot_confusion_matrix(y_true, y_pred, output_path)
    plot_roc_curve(y_true, y_pred, output_path)
    plot_precision_recall_curve(y_true, y_pred, output_path)
