from treenode import TreeNode
from objective.simple import Objective
from util.util import profile
import numpy as np

Y = '_y_column'


class CustomClassifier:
    trees = []
    obj = Objective()

    def __init__(self, tree_count=2, split_count=3):
        self.tree_count = tree_count
        self.split_count = split_count

    @profile
    def _calc_split(self, data):
        rv = {"obj": float("inf")}
        for factor in [x for x in self.data_cols if x not in [self.target, Y]]:
            f_idx = self.data_cols.index(factor)
            values = np.unique(data[:, f_idx])
            if len(values) > 1:
                for value in values:
                    lo_data = data[data.T[f_idx] <= value, :]
                    hi_data = data[data.T[f_idx] > value, :]
                    if hi_data.shape[0] > 0:
                        obj_v, lo_w, hi_w = self.obj.get_objective(lo_data, hi_data, self.idx_target, self.idx_y)
                        if obj_v < rv.get('obj'):
                            rv = {'obj': obj_v, 'factor': factor, 'value': value, 'lo_data': lo_data, 'hi_data': hi_data,
                                  'lo_w': lo_w,
                                  'hi_w': hi_w}
        return TreeNode(**{k: v for k, v in rv.items() if k in TreeNode.fieldnames}), rv

    @profile
    def _get_node(self, data, level=1):
        if level > self.split_count:
            node = TreeNode(level=level)
        else:
            node, meta = self._calc_split(data)
            node.level = level
            if not node.is_leaf():
                node.lo_branch = self._get_node(meta.get('lo_data'), level=level + 1)
                node.lo_branch.w = meta.get('lo_w')
                node.hi_branch = self._get_node(meta.get('hi_data'), level=level + 1)
                node.hi_branch.w = meta.get('hi_w')
        return node

    def _calc_y(self, record):
        return sum([t.evaluate(record) for t in self.trees])

    def _set_data_dict(self, df, target):
        self.target = target
        self.data_cols = cols = list(df.columns)
        self.idx_target = cols.index(target)
        self.idx_y = cols.index(Y)

    @profile
    def _build_next_tree(self, df, target):
        df[Y] = df.apply(lambda rec: self._calc_y(rec), axis=1)
        self._set_data_dict(df, target)
        return self._get_node(df.values)

    def _print_data_amount(self, data):
        for col in data.columns:
            print("{}: {}".format(col, len(data[col].unique())))

    def fit(self, df, target='status'):
        self._print_data_amount(df)
        while len(self.trees) < self.tree_count:
            print("New tree")
            self.trees.append(self._build_next_tree(df, target))

    def predict(self, data):
        return data.apply(lambda row: self._calc_y(row), axis=1 )
