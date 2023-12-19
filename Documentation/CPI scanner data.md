# CPI Scanner data

CPI scanner data gives a list of all products in a supermarket's inventory over a certain
period of time. The advantage is that it contains an integral view of the inventory of a
supermarket at a certain time. The disadvantages can be that this view is not real-time;
there may be a delay in when the supermarket delivers the product inventory to the NSI.
Also, supermarkets can decide not to deliver scanner data or to stop delivering scanner
data. The sample of supermarkets may therefore not be representative nor complete. In addition, not all the products in the inventory will be labeled.

- Does it really contain all the products in the inventory of a supermarket or is it a sample?
- Do we have scanner data on an individual level? Per bank-account or per receipt?

## CPI scanner data pipeline

The CPI scanner data pipeline:

1. Convert CPI csv to parquet files. Two kind of files:
   - Files with receipt texts
   - Files with supermarket revenue
2. Pre-process CPI data:
   1. Combine separate files (if any)
   2. Filter out unused columns
   3. Rename the columns to a standardized format
   4. Unify the length of the COICOP numbers (6 digits, prepend zeroes)
   5. Split the month_year column in two separate columns for month and year
   6. Add a unique product id: a hash based on the receipt text
   7. Split the COICOP number into columns for each COICOP level.
   8. Get the number of products in each column
3. Extract features from the receipt text using:
   - CountVectorizer
   - TFIDFVectorizer
   - Word embeddings: spacy NL embeddings.
