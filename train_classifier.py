from ssi.feature_extraction import FeatureExtractorType
from ssi.train_model import ModelType, train_model_with_feature_extractors
import argparse


def main(args):
    feature_extractors = [FeatureExtractorType(feature_extractor)
                          for feature_extractor in args.feature_extractors] if args.feature_extractors else [FeatureExtractorType.spacy_nl_md]

    train_model_with_feature_extractors(args.input_filename,
                                        args.receipt_text_column,
                                        args.coicop_column,
                                        feature_extractors,
                                        ModelType(args.model),
                                        args.test_size,
                                        args.output_filename)


if __name__ == "__main__":
    feature_extractor_choices = [feature_extractor_type.value
                                 for feature_extractor_type in FeatureExtractorType]
    model_choices = [model_type.value for model_type in ModelType]

    parser = argparse.ArgumentParser(description='Train a COICOP classifier')
    parser.add_argument("-i", "--input-filename", type=str,
                        required=True, help="The input filename in parquet format")
    parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text",
                        help="The column name containing the receipt text")
    parser.add_argument("-c", "--coicop-column", type=str, default="coicop",
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
    args = parser.parse_args()
    main(args)
