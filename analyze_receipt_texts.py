from ssi.analysis.text_analysis import analyze_supermarket_receipts
from ssi.constants import Constants
import argparse

parser = argparse.ArgumentParser(description='Analyze supermarket receipts')
parser.add_argument("-i", "--input-filename", type=str,
                    required=True, help="The input filename in parquet format")
parser.add_argument("-s", "--supermarket-name", type=str,
                    required=True, help="The supermarket name")
parser.add_argument("-o", "--output-directory", type=str,
                    required=True, help="The output directory")
parser.add_argument("-y", "--year-column", type=str, default="year",
                    help="The column name containing the year")
parser.add_argument("-m", "--month-column", type=str, default=Constants.YEAR_MONTH_COLUMN,
                    help="The column name containing the month")
parser.add_argument("-r", "--receipt-text-column", type=str, default=Constants.RECEIPT_TEXT_COLUMN,
                    help="The column name containing the receipt text")
args = parser.parse_args()

analyze_supermarket_receipts(args.input_filename, args.supermarket_name,
                             args.output_directory, args.year_column, args.month_column, args.receipt_text_column)
