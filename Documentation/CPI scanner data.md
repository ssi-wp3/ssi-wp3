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
4. Train an ML classifier on the extracted features

Note: Lidl uses EAN name from the revenue file for receipt text, whereas Plus uses a
separate file for the receipt text and the revenue. Provide one unified file formate
for both supermarkets. Check code of @HBS.

## Analysis of COICOP levels/Product texts

- CBS uses an old COICOP classification from before 2018. Which COICOP classification?
- Do all COICOP numbers have the same length?
  - For a five level COICOP classification we expect 6 digits; level one has two digits, classes 1-9 starting with a zero, i.e. 01-09.
  - Give a value counts of the COICOP number lengths.
  - Check how many products cannot be extended to length 6.
  - Check how many unique COICOP numbers there are in the dataset.
  - Do any of the COICOP numbers already start with a 0?
- Are EAN numbers unique?
  - Do they always belong to the same COICOP category?
  - Do they always have the same (or similar) receipt text?
  - Create a "transition" matrix, which EAN goes from which COICOP level to which?
  - Do we see most transitions from/to the 99 or "other" category?
  - Do transitions appear over time? For example 99 -> specific COICOP level
  - Chord diagram of EAN numbers?
  - Parallel coordinate plot of EAN numbers over time?
- Count number of unique products per COICOP level (level 1 until 5).
  - EAN numbers
  - Unique receipt texts.
  - Sunburst plots; gives hierarchical plot of product hierarchies
  - Number of unique texts per time unit (year/year-month)
  - Number of unique texts per time unit and COICOP category
  - Chord diagram of product hashes
  - Parallel coordinate plot of product hashes
- Check the types of product descriptions in each COICOP level:
  - Manually by looking a some random product descriptions
  - By creating word clouds
- Do receipt text/product descriptions change category?

## Analysis of Feature Space

## Evaluation of the ML performance

- Train time evaluation; how well does the algorithm perform?
  - Learning curve!
- Evaluation in production; how well does the algorithm perform over time?
  - F1 of algorithm trained on one time period, tested on other time periods. GroupKFold on years?
- Should we train our own embedding?
