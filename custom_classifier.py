from treenode import TreeNode
import numpy as np

Y = '_y_column'


class CustomClassifier:
    trees = []
    lambda_1 = 1
    weights = {0.: 1., 1.: 1.}

    def __init__(self, tree_count=5, split_count=5):
        self.tree_count = tree_count
        self.split_count = split_count

    def _calc_split(self, data, target):
        rv = {"obj": float("inf")}
        for factor in [x for x in data.columns if x not in [target, Y]]:
            values = data[factor].unique()
            if len(values) > 1:
                print("{} ({})".format(factor, len(values)))
                for value in values:
                    print(".", end="", flush=True)
                    lo_data = data[data[factor] <= value]
                    hi_data = data[data[factor] > value]
                    if hi_data.shape[0] > 0:
                        obj, lo_w, hi_w = self._get_obj(self._get_g(lo_data, target), self._get_h(lo_data, target),
                                                        self._get_g(hi_data, target), self._get_h(hi_data, target))
                        if obj < rv.get('obj'):
                            rv = {'obj': obj, 'factor': factor, 'value': value, 'lo_data': lo_data, 'hi_data': hi_data,
                                  'lo_w': lo_w,
                                  'hi_w': hi_w}
                print("")
        return TreeNode(**{k: v for k, v in rv.items() if k in TreeNode.fieldnames}), rv

    def _get_node(self, data, target, level=1):
        print('level=', level)
        if level > self.split_count:
            node = TreeNode(level=level)
        else:
            node, meta = self._calc_split(data, target)
            node.level = level
            if not node.is_leaf():
                node.lo_branch = self._get_node(meta.get('lo_data'), target, level=++level)
                node.lo_branch.w = meta.get('lo_w')
                node.hi_branch = self._get_node(meta.get('hi_data'), target, level=++level)
                node.hi_branch.w = meta.get('hi_w')
        print('factor=', node.factor, 'value=', node.value)
        return node

    def _calc_y(self, record):
        return sum([t.evaluate(record) for t in self.trees])

    def _build_next_tree(self, data, target):
        data[Y] = data.apply(lambda rec: self._calc_y(rec), axis=1)
        return self._get_node(data, target)

    def _g_func(self, val_target, val_y):
        return self.weights.get(val_target) * 2 * (val_y - val_target)

    def _h_func(self, val_target):
        return self.weights.get(val_target) * 2

    def _get_g(self, data, target):
        return sum(np.vectorize(self._g_func)(data[target], data[Y]))

    def _get_h(self, data, target):
        return sum(np.vectorize(self._h_func)(data[target]))

    def _get_obj(self, g_lo, h_lo, g_hi, h_hi):
        return -0.5 * (g_lo * g_lo / (h_lo + self.lambda_1) + g_hi * g_hi / (h_hi + self.lambda_1)), \
               self._get_w(g_lo, h_lo), \
               self._get_w(g_hi, h_hi)

    def _get_w(self, g, h):
        return -g / (h + self.lambda_1)

    def _print_data_amount(self, data):
        for col in data.columns:
            print("{}: {}".format(col, len(data[col].unique())))

    def fit(self, data, target='status'):
        self._print_data_amount(data)
        while len(self.trees) <= self.tree_count:
            print("New tree")
            self.trees.append(self._build_next_tree(data, target))

    def predict(self, data):
        return data.apply(lambda row: self._calc_y(row), axis=1 )
