#!/bin/bash
python generate_synthetic_data.py -n 1000 -dr 2016-2019 -o Omzet_supermarket1_synthetic_data1.parquet -s 995001 
python generate_synthetic_data.py -n 500 -dr 2019-2020 -o Omzet_supermarket1_synthetic_data2.parquet -s 995001
python generate_synthetic_data.py -n 1000 -dr 2021-2023 -o Omzet_supermarket1_synthetic_data3.parquet -s 995001
python generate_synthetic_data.py -n 3000 -dr 2018-2023 -o Omzet_supermarket2_synthetic_data1.parquet -s 995002
python generate_synthetic_data.py -n 2000 -dr 2016-2020 -o Omzet_supermarket3_synthetic_data1.parquet -s 995003
python generate_synthetic_data.py -n 2000 -dr 2020-2023 -o Omzet_supermarket3_synthetic_data2.parquet -s 995003
