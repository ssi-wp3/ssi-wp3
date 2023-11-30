from sklearn.metrics import confusion_matrix
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calculate_and_plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(matrix)


def plot_confusion_matrix(confusion_matrix: np.array, title: str = 'Confusion matrix'):
    # Create a heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Show the plot
    plt.show()