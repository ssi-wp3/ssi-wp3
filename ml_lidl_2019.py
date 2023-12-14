import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from gama import GamaClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

#DATA_DIR = os.path.join("..", "data")
DATA_DIR = "/data/projecten/ssi/data/features"

COL_MAPPING = {
  "coicop_division"     : "coicop_1",
  "coicop_group"        : "coicop_2",
  "coicop_class"        : "coicop_3",
  "coicop_subclass"     : "coicop_4",
  "month"               : "year_month",
  "features_spacy_nl_md": "word_embeds",
  #"features_count_vectorizer": "word_embeds",
}

Y_COLUMN = "coicop_1"
X_COLUMN = "word_embeds"

def get_X_y(dataset):
  X = dataset[X_COLUMN]
  X = pd.DataFrame(X.tolist())
  y = dataset[Y_COLUMN]
  return X, y

def fit_gama(clf, train, test, i):
  # split X and y
  X_train, y_train = get_X_y(train)
  X_test, y_test = get_X_y(test)

  print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")  
  print(X_train.columns)
  print(y_train.value_counts())

  print(f"X_test shape: {X_test.shape}, y_test: {y_test.shape}")  
  print(X_test.columns)
  print(y_test.value_counts())

  # train test split: train: 2022, test: 2023
  cv_scores = cross_validate(clf, X_train, y_train, cv=5, scoring=("balanced_accuracy", "f1_macro"), return_train_score=True)
  print(cv_scores)

  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  print(classification_report(y_test, y_pred))
  print(f"balanced acc: ", balanced_accuracy_score(y_test, y_pred))
  print(f"f1          : ", f1_score(y_test, y_pred, average="macro"))


  # confusion matrix
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  plt.figure(figsize=(10, 10))
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  plt.savefig(f"conf_matrix_{i}.png")


if __name__ == "__main__":
  df = pd.read_parquet(os.path.join(DATA_DIR, "ssi_lidl_spacy_nl_md_features.parquet"))
  #df = pd.read_parquet(os.path.join(DATA_DIR, "ssi_lidl_count_vectorizer_features.parquet"))
  df = df.rename(COL_MAPPING, axis=1)
  print(df.head())

  # extract year column from year_month
  #df["year"] = df["year_month"].str.get(0)   \
  #             .add(df["year_month"].str.get(1)) \
  #             .add(df["year_month"].str.get(2)) \
  #             .add(df["year_month"].str.get(3)) \
  #             .astype("uint16")

  df["year"] = df["year_month"].str[:4]

  # df.iloc[df[df["year"] == 2019]["ean_name"].drop_duplicates().index]
  train_df = df[df["year"] == "2019"]
  test_2020_df = df[df["year"] == "2020"]
  test_2021_df = df[df["year"] == "2021"]
  test_2022_df = df[df["year"] == "2022"]
  test_2023_df = df[df["year"] == "2023"]
    
  fit_gama(clf=LogisticRegression(),
           train=train_df,
           test=test_2020_df,
           i="2020")

  fit_gama(clf=LogisticRegression(),
           train=train_df,
           test=test_2021_df,
           i="2021")

  fit_gama(clf=LogisticRegression(),
           train=train_df,
           test=test_2022_df,
           i="2022")

  fit_gama(clf=LogisticRegression(),
           train=train_df,
           test=test_2023_df,
           i="2023")
