import pandas as pd
from sklearn.model_selection import learning_curve
from .hyper_params.hyper_params import FeatureExtractorType, ModelType
from .hyper_params.pipeline import pipeline_with
from ..preprocessing.combine_unique_values import drop_empty_receipts, drop_unknown
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

data_directory = os.getenv('data_directory', '.')
features_directory = os.path.join(data_directory, 'features')
output_directory = os.path.join(
    data_directory, 'machine_learning/learning_curve')

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Learning Curve Script')
# Add arguments to the parser
parser.add_argument('-i', '--input-filename', type=str, default=os.path.join(data_directory, 'ssi_hf_labse_unique_values.parquet'),
                    help='Path to the input dataset parquet file')
parser.add_argument('-o', '--output-filename', type=str, default='learning_curve_results',
                    help='Filename for the learning curve results')
parser.add_argument('-f', '--feature-extractor-type', type=FeatureExtractorType,
                    choices=list(FeatureExtractorType), default=FeatureExtractorType.count_vectorizer, help='Type of feature extractor to use')
parser.add_argument('-m', '--model-type', type=ModelType,
                    choices=list(ModelType), default=ModelType.logistic_regression, help='Type of model to use')
parser.add_argument('-rc', '--receipt-text-column', type=str, default='receipt_text',
                    help='Name of the input column in the dataset')
parser.add_argument('-l', '--label-column', type=str, default='coicop_number',
                    help='Name of the output column in the dataset')
parser.add_argument('-k', '--number-of-folds', type=int,
                    default=5, help='Number of folds for cross-validation')
parser.add_argument('-e', '--engine', type=str,
                    default='pyarrow', help='Parquet engine to use')
parser.add_argument('-d', '--delimiter', type=str, default=';',
                    help='Delimiter to use for the CSV file')
parser.add_argument('-k', '--keep-unknown',
                    action='store_true', help='Keep unknown labels')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Parse the command-line arguments
args = parser.parse_args()

# Load the dataset from the parquet file
dataframe = pd.read_parquet(args.input_filename, engine=args.engine)
dataframe = drop_empty_receipts(
    dataframe, args.receipt_text_column)

if not args.keep_unknown:
    dataframe = drop_unknown(dataframe, args.label_column)

# Split the data into input (X) and output (y)
X = dataframe[args.receipt_text_column]
y = dataframe[args.label_column]

# Create a classification pipeline
pipeline_params = {
    'vectorizer__analyzer': 'char',
    'vectorizer__lowercase': True,
    'vectorizer__max_df': 0.8863890037284232,
    'vectorizer__max_features': 8766,
    'vectorizer__ngram_range': (1, 3),
    'clf__fit_intercept': True,
    'clf__max_iter': 1000,
    'clf__penalty': 'l2',
}

pipeline = pipeline_with(
    args.feature_extractor_type, args.model_type)
pipeline.set_params(**pipeline_params)


# Generate the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=args.number_of_folds)

input_filename = os.path.splitext(os.path.basename(args.input_filename))[0]

# Save the scores including the training sizes to a CSV file
# Save scores for each fold
learning_curve_results = pd.DataFrame({
    'train_sizes': train_sizes
})

for i in range(args.number_of_folds):
    learning_curve_results[f'train_scores_{i}'] = train_scores[:, i]
    learning_curve_results[f'test_scores_{i}'] = test_scores[:, i]
learning_curve_results.to_csv(os.path.join(output_directory,
                                           f'{args.output_filename}_{input_filename}.csv'),
                              sep=args.delimiter,
                              index=False)


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

plt.savefig(os.path.join(output_directory,
            f'{args.output_filename}_{input_filename}.png'))
