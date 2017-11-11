class TreeNode:
    fieldnames = ['factor', 'level', 'value', 'lo_branch', 'hi_branch', 'w']
    factor = None
    level = None
    value = None
    lo_branch = None
    hi_branch = None
    w = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k,v)

    def is_leaf(self):
        return self.factor is None

    def evaluate(self, data_series):
        if self.is_leaf():
            return self.w
        else:
            branch = self.lo_branch if data_series[self.factor] <= self.value else self.hi_branch
            return branch.evaluate(data_series)