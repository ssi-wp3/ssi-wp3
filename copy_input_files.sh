#!/bin/sh
export raw_directory=$data_directory/preprocessing/00-raw
export cleaned_directory=$data_directory/preprocessing/01-cleaned

#check if directories exist, otherwise create them
if [ ! -d $raw_directory ]; then
  mkdir -p $raw_directory
fi

if [ ! -d $cleaned_directory ]; then
  mkdir -p $cleaned_directory
fi


export input_directory=~/shares/productie/primair/CPI/Output/BudgetOnderzoek/HBS2

# Plus and Lidl data
cp $input_directory/99EanOutputBestand/Omzet*.csv $raw_directory 
cp $input_directory/99EanOutputBestand/Output*.csv $raw_directory
cp $input_directory/99EanOutputBestand/KassabonPlus_va_202201_Prd.csv $raw_directory/receipts_plus.csv

# AH data
cp $input_directory/08BestandAlbertHeijn/EANOmzet_202107_202312.csv $raw_directory/OmzetEansCoicopsAH_202107_202312.csv
cp $input_directory/08BestandAlbertHeijn/Copy*.xlsx $raw_directory/receipts_ah.xlsx

#Jumbo data
cp $input_directory/09BestandenJumbo/JumboEANomzet_202212_202402.csv $raw_directory/OmzetEansCoicopsJumbo_202212_202402.csv
cp $input_directory/09BestandenJumbo/part*.csv $cleaned_directory
