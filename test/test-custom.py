import unittest
import pandas as pd
from custom_classifier import CustomClassifier


def test_classifier():
  df = pd.read_csv("../data/merchant-11/train-custom-xgboost.csv")
  cc = CustomClassifier()
  cc.fit(df, 'status')


if __name__ == '__main__':
  unittest.main()


