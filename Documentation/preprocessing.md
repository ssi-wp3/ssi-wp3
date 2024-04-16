# Preprocessing

Before being able to be use in the analysis, feature extraction, and machine
learning stages, input files need to be preprocessed and standardized. After this
process, the files for all stores will have the same columns and include the
necessary information that is expected in the steps further onward.

The preprocessing described here is specific to the files that are used at
Statistics Netherlands. At Statistic Netherlands, we use the CPI scanner data as
a source of input. In addition, we have a file with receipt texts for almost all
supermarkets. In the preprocessing steps, these file are cleaned and combined
into one large file per store that can be used further on. While the input may
differ at other NSIs, most likely, the files will contain similar information.
It will most probably be possible to write a NSI specific preprocessing stage
that delivers output files with the same columns.

# TODO describe output file structure in more detail

For Statistics Netherlands, the preprocessing stage consists of the following
steps:

1. Cleaning the files (`luigi_preprocess_csvs.sh`)
2. Converting the files to the Apache Parquet file format (`luigi_convert_receipts.sh`)
3. Combine the separate files if there are more than one (`luigi_combine_files.sh` carries out step 2 & 3 for the CPI files)
4. Preprocess the file content (`luigi_preprocess_files.sh`)
5. Join the CPI files with the files containing the receipt texts (`luigi_add_receipts.sh`)
