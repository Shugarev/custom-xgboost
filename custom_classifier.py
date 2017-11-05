Y = '_y_column'

class CustomClassifier:

  trees = []
  lambda_1 = 1
  weights = {0.: 1., 1.: 1.}

  class TreeNode:
    factor = None
    level = 0
    value = None
    lo_branch = None
    hi_branch = None

  def __init__(self, tree_count=5, split_count=5):
    self.tree_count = tree_count
    self.split_count = split_count

  def _calc_split(self, data, target):
    rv = {"obj": 1}  # Return value. результат же всегда меньше 0?
    for factor in [x for x in data.columns if x != target]:
      values = data[factor].unique()
      for value in values:
        lo_data = data[data[factor] <= value]
        hi_data = data[data[factor] > value]
        if hi_data.shape[0] > 0:
          obj, w_lo, w_hi = self._get_obj(self._get_g(lo_data, target), self._get_h(lo_data, target),
                              self._get_g(hi_data, target), self._get_h(hi_data, target))
          if obj < rv['obj']:
            rv['obj'] = obj
            rv['factor'] = factor
            rv['value'] = value
            rv['lo_data'] = lo_data
            rv['hi_data'] = hi_data
            rv['w_lo'] = w_lo
            rv['w_hi'] = w_hi
    rv['lo_data'] = lo = rv['lo_data'].copy()
    rv['hi_data'] = hi = rv['hi_data'].copy()
    lo[Y] = rv['w_lo']
    hi[Y] = rv['w_hi']
    return rv.get('factor'), rv.get('value'), rv.get('lo_data'), rv.get('hi_data')

  def _get_node(self, data, target, level=1):
    node = self.TreeNode()
    if Y not in data.columns:
      data[Y] = 0
    node.factor, node.value, data_lo, data_hi = self._calc_split(data, target)
    print('factor=', node.factor, 'value=', node.value)
    if level < self.split_count:
      node.lo_branch = self._get_node(data_lo, target, level=++level)
      node.hi_branch = self._get_node(data_hi, target, level=++level)
    return node

  def _build_next_tree(self, data, target):
    return self._get_node(data, target)

  def _get_g(self, data, target):
    return sum(data.apply(lambda row: self.weights.get(row[target]) * 2 * (row[Y] - row[target]), axis=1))

  def _get_h(self, data, target):
    return sum(data.apply(lambda row: self.weights.get(row[target]) * 2, axis=1))

  def _get_obj(self, g_lo, h_lo, g_hi, h_hi):
    return -0.5 * (g_lo * g_lo / (h_lo + self.lambda_1) + g_hi * g_hi / (h_hi + self.lambda_1)), \
           self._get_w(g_lo, h_lo), \
           self._get_w(g_hi, h_hi)

  def _get_w(self, g, h):
    return -g / (h + self.lambda_1)

  def fit(self, data, target='status'):
    while len(self.trees) <= self.tree_count:
      self.trees.append(self._build_next_tree(data, target))

  def predict(self, data):
    pass