import unittest
import pandas as pd
from custom_classifier import CustomClassifier


class CustomTests(unittest.TestCase):
    def test_classifier_fit(self):
        df = pd.read_csv("../data/merchant-11/train-custom-xgboost.csv")
        # df = pd.read_csv("../data/merchant-11/train.csv")
        bad_columns = ['id', 'date_only']
        df = df[[x for x in df.columns if x not in bad_columns]]
        cc = CustomClassifier()
        cc.fit(df, 'status')
        print(cc.predict(df))


if __name__ == '__main__':
    unittest.main()
