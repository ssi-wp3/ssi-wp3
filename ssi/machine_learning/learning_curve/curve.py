import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier
import argparse
import matplotlib.pyplot as plt
import os

data_directory = os.getenv('data_directory', '.')
features_directory = os.path.join(data_directory, 'features')
output_directory = os.path.join(
    data_directory, 'machine_learning/learning_curve')

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Learning Curve Script')
# Add arguments to the parser
parser.add_argument('--dataset', type=str, default=os.path.join(data_directory, 'ssi_hf_labse_unique_values.parquet'),
                    help='Path to the input dataset parquet file')
parser.add_argument('--output', type=str, default='learning_curve_results.csv',
                    help='Filename for the learning curve results')
parser.add_argument('--input_column', type=str, default='receipt_text',
                    help='Name of the input column in the dataset')
parser.add_argument('--output_column', type=str, default='coicop_number',
                    help='Name of the output column in the dataset')

# Parse the command-line arguments
args = parser.parse_args()

# Load the dataset from the parquet file
data = pd.read_parquet(args.dataset)

# Split the data into input (X) and output (y)
X = data[args.input_column]
y = data[args.output_column]

# Rest of the code...
# Load the dataset from the parquet file
data = pd.read_parquet('/path/to/your/dataset.parquet')

# Split the data into input (X) and output (y)
X = data['receipt_text']
y = data['coicop_number']

# Create an instance of the SGDClassifier
classifier = SGDClassifier()

# Generate the learning curve
train_sizes, train_scores, test_scores = learning_curve(classifier, X, y, cv=5)

# Calculate the mean and standard deviation of the training and test scores
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure()
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
         label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
         label='Cross-validation score')

plt.legend(loc='best')
plt.show()
