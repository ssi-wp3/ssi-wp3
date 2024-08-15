import duckdb
import argparse
import os

data_directory = os.getenv("data_directory", default=".")
db_directory = os.path.join(data_directory, "string_matching")
os.makedirs(db_directory, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename", type=str, default=os.path.join(
    db_directory, "coicop.db"), help="The output database file.")
parser.add_argument("-t", "--receipt-text-table", type=str, default="receipt_text",
                    help="The name of the table containing the receipt texts.")
parser.add_argument("-r", "--receipt-text-column", type=str, default="receipt_text",
                    help="The name of the column containing the receipt texts.")
parser.add_argument('receipt_texts', nargs=argparse.REMAINDER)

args = parser.parse_args()

con = duckdb.connect(args.input_filename)

print("-"*80)
for query_string in args.receipt_texts:
    print(con.sql(f"""select index, {args.receipt_text_column}, score from 
            (
            SELECT *, fts_main_{args.receipt_text_table}.match_bm25(
                index,
                '{query_string}',
                fields := '{args.receipt_text_column}'
            ) AS score
            FROM {args.receipt_text_table}
            )
            where score is not null
            order by score desc;        
            """).fetchmany(5))
    print("-"*80)
