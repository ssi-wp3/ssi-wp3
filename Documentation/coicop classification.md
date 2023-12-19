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

Labeled receipts can be used as testing material for both the OCR as the COICOP classification. OCR sometimes detects letters/words wrong. To test the effects of this

## String matching

How well does it work?

Evaluate the effect of OCR distortions on COICOP classification.

Can we simulate OCR distortions to texts?

Evaluate how often small string distances change the COICOP label:

- Use bootstrapping?
