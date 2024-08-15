import duckdb
import argparse
import os

data_directory = os.getenv("data_directory", default=".")
db_directory = os.path.join(data_directory, "string_matching")
os.makedirs(db_directory, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename", type=str, required=True,
                    help="The parquet file containing the receipt texts and their corresponding COICOP labels.")
parser.add_argument("-o", "--output-filename", type=str, default=os.path.join(
    db_directory, "coicop.db"), help="The output database file.")
parser.add_argument("-t", "--receipt-text-table", type=str, default="receipt_text",
                    help="The name of the table containing the receipt texts.")
parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text",
                    help="The name of the column containing the receipt texts.")
args = parser.parse_args()

con = duckdb.connect(args.output_filename)
con.sql("""INSTALL fts;
        LOAD fts;
""")
con.sql(f"""drop table if exists {args.receipt_text_table};
            create table {args.receipt_text_table} as select *
            from read_parquet('{args.input_filename}');
        """)
con.sql(f"""PRAGMA create_fts_index({args.receipt_text_table}, 'index', {args.receipt_text_column},
        stemmer='dutch', stop_words='dutch'); 
""")
