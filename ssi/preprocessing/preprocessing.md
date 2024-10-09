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

## Copying files into the 00-raw folder

The preprocessing stage starts with the csv files in the 00-raw folder. The
files in this folder are the original CPI files. The CPI files need to be
copied manually into the 00-raw folder. Most CPI files need to be cleaned
first, before they can be used in the preprocessing stage. To start the preprocessing
first the `preprocessing` directory and the `preprocessing/00-raw` directory need to be
created. This can be done by running the following commands:

```cli
cd /path/to/data_directory
mkdir preprocessing
cd preprocessing
mkdir 00-raw
```

For the preprocessing scripts to work, furthermore the  `$data_directory` environment variable 
needs to be set. This can be done by running the following command:

```cli
export data_directory=/path/to/data_directory
```

## Preprocessing for Statistic Netherlands

For Statistics Netherlands, the preprocessing stage consists of the following
steps:

1. Cleaning the files (`luigi_preprocess_csvs.sh`)
2. Converting the files to the Apache Parquet file format (`luigi_convert_receipts.sh`)
3. Combine the separate files if there are more than one (`luigi_combine_files.sh` carries out step 2 & 3 for the CPI files)
4. Preprocess the file content (`luigi_preprocess_files.sh`)
5. Join the CPI files with the files containing the receipt texts (`luigi_add_receipts.sh`)

## Directory structure
