from ssi.feature_extraction.feature_extraction import FeatureExtractorType
from ssi.train_model import ModelFactory, train_model_with_feature_extractors
from ssi.label_extractor import LabelExtractorFactory
import argparse


def main(args):
    label_extractor = LabelExtractorFactory().get_label_extractor_for_model(
        args.model, args.coicop_column)
    feature_extractors = [FeatureExtractorType(feature_extractor)
                          for feature_extractor in args.feature_extractors] if args.feature_extractors else [FeatureExtractorType.spacy_nl_md]

    train_model_with_feature_extractors(args.input_filename,
                                        args.receipt_text_column,
                                        args.coicop_column,
                                        label_extractor,
                                        feature_extractors,
                                        args.model,
                                        args.test_size,
                                        args.output_filename,
                                        args.number_of_jobs,
                                        args.verbose)


if __name__ == "__main__":
    feature_extractor_choices = [feature_extractor_type.value
                                 for feature_extractor_type in FeatureExtractorType]
    model_choices = ModelFactory().model_names

    parser = argparse.ArgumentParser(description='Train a COICOP classifier')
    parser.add_argument("-i", "--input-filename", type=str,
                        required=True, help="The input filename in parquet format")
    parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text",
                        help="The column name containing the receipt text")
    parser.add_argument("-c", "--coicop-column", type=str, default="coicop_number",
                        help="The column name containing the COICOP code")
    parser.add_argument("-f", "--feature-extractors", type=str, nargs="+", default=[],
                        choices=feature_extractor_choices,
                        help="Feature extractors to use")
    parser.add_argument("-m", "--model", type=str, choices=model_choices, required=True,
                        help="The model to use")
    parser.add_argument("-o", "--output-filename", type=str,
                        required=True, help="The output filename for the pipeline")
    parser.add_argument("-t", "--test-size", type=float, default=0.2,
                        help="The test size for the train/test split")
    parser.add_argument("-n", "--number-of-jobs", type=int, default=-1,
                        help="The number of jobs to use")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    main(args)
