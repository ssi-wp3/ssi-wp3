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
parser.add_argument('-i', '--input-filename', type=str, default=os.path.join(data_directory, 'ssi_hf_labse_unique_values.parquet'),
                    help='Path to the input dataset parquet file')
parser.add_argument('-o', '--output-filename', type=str, default='learning_curve_results.csv',
                    help='Filename for the learning curve results')
parser.add_argument('-rc', '--receipt-text-column', type=str, default='receipt_text',
                    help='Name of the input column in the dataset')
parser.add_argument('-l', '--label-column', type=str, default='coicop_number',
                    help='Name of the output column in the dataset')
parser.add_argument('-k', '--number-of-folds', type=int,
                    default=5, help='Number of folds for cross-validation')
parser.add_argument('-e', '--engine', type=str,
                    default='pyarrow', help='Parquet engine to use')

# Parse the command-line arguments
args = parser.parse_args()

# Load the dataset from the parquet file
data = pd.read_parquet(args.input_filename, engine=args.engine)

# Split the data into input (X) and output (y)
X = data[args.receipt_text_column]
y = data[args.label_column]

# Create an instance of the SGDClassifier
classifier = SGDClassifier()

# Generate the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    classifier, X, y, cv=args.number_of_folds, exploit_incremental_learning=True)

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

input_filename = os.path.splitext(os.path.basename(args.input_filename))[0]
plt.savefig(os.path.join(output_directory,
            f'learning_curve_sgd_{input_filename}.png'))
