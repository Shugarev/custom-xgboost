import unittest
import pandas as pd
from util.util import TIME, COUNT
from custom_classifier import CustomClassifier


class CustomTests(unittest.TestCase):

    @staticmethod
    def _print_perf_info(info):
        print('{:20}|{:7}|{:12}'.format('  function', '  count', '  time'))
        for fn, data in info.items():
            print('{:20}|{:7d}|{:8.4f}'.format(fn, data.get(COUNT), data.get(TIME)))

    def test_classifier_fit(self):
        # df = pd.read_csv("../data/merchant-11/train-custom-xgboost.csv")
        # df = pd.read_csv("../data/merchant-11/train.csv")
        # df = pd.read_csv("../data/merchant-11/train5k.csv")
        # df = pd.read_csv("../data/merchant-11/train10k.csv")
        df = pd.read_csv("../data/merchant-11/train100k.csv")
        bad_columns = ['id', 'date_only']
        df = df[[x for x in df.columns if x not in bad_columns]]
        cc = CustomClassifier()
        cc.fit(df, 'status')
        print("-------- perf info ---------")
        self._print_perf_info(cc._perf_info)
        # print(cc.predict(df))


if __name__ == '__main__':
    unittest.main()

