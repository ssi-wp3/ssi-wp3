from .results import boxplot_files
import argparse

parser = argparse.ArgumentParser(
    description='Create a boxplot for the given metrics.')
parser.add_argument('-f', '--files', metavar='files', type=str, nargs='+',
                    help='List of file names.')
parser.add_argument('-d', '--delimiter', type=str, default=';',
                    help='Delimiter to use for reading the files.')
parser.add_argument('-p', '--plot_filename', type=str, default='boxplot.png',
                    help='Filename for the boxplot.')
parser.add_argument('-m', '--metric_name', type=str, default='metric',
                    help='Name of the column containing the metric names.')
parser.add_argument('-g', '--group_name', type=str, default='store',
                    help='Name of the column containing the group names.')
args = parser.parse_args()

boxplot_files(args.files, args.delimiter, args.plot_filename,
              args.metric_name, args.group_name)
