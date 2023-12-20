# COICOP classification

Three scenarios:

1. Use of machine learning
2. String-matching
3. Manual search

## Machine learning

Trained on:

- labeled receipts
- CPI scanner data

### Labeled receipts

Products on receipts can be manually labeled with their COICOP category. The labeled data can then be used to train an ML classifier. However, a lot of receipts need to be collected to be able to have a representative sample of all COICOP categories:

- Is there a way to calculate the number of receipts we need to collect?
- Can we use the CPI sales volumes to say something about that?
- Can we say something about individual behavior? How often do individuals buy the same products.
- Receipts say something about real-world behavior on an individual or household level, wherease CPI data says something about collective behavior. Or do we have scanner data per receipt?

Labeled receipts can be used as testing material for both the OCR as the COICOP
classification. OCR sometimes detects letters/words wrong. To test the effects of these
distortions on COICOP classification we need real-world data, i.e. labeled receipts.

### CPI scanner data

CPI scanner data gives a list of all products in a supermarket's inventory over a certain
period of time. The advantage is that it contains an integral view of the inventory of a
supermarket at a certain time. The disadvantages can be that this view is not real-time;
there may be a delay in when the supermarket delivers the product inventory to the NSI.
Also, supermarkets can decide not to deliver scanner data or to stop delivering scanner
data. The sample of supermarkets may therefore not be representative nor complete. In addition, not all the products in the inventory will be labeled.

More information about the CPI scanner data can be found [here](./CPI%20scanner%20data.md)

## String matching

How well does it work?

Evaluate the effect of OCR distortions on COICOP classification.

Can we simulate OCR distortions to texts?

Evaluate how often small string distances change the COICOP label:

- Use bootstrapping?
