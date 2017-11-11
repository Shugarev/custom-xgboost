import unittest
from treenode import TreeNode


class NodeTests(unittest.TestCase):
    def test_node_empty(self):
        n = TreeNode()
        self.assertEqual(True, n.is_leaf())

    def test_node_init(self):
        n = TreeNode(factor="amount", level=3, value=5, w=-0.1434, lo_branch="empty", hi_branch="empty")
        self.assertEqual("amount", n.factor)
        n = TreeNode(**{'obj': 33.33})
        self.assertEqual(33.33, n.obj)

    def test_node_evaluate(self):
        n_low = TreeNode(w=-0.1434)
        n_hi = TreeNode(w=-0.456)
        n_week = TreeNode(factor="day_of_week", value=3, lo_branch=n_low, hi_branch=n_hi)
        n_bigamount = TreeNode(w=0.2)
        n_root = TreeNode(factor="amount", value=5, lo_branch=n_week, hi_branch=n_bigamount)
        data1 = {'amount': 2, 'day_of_week': 2}
        data2 = {'amount': 8, 'day_of_week': 2}
        self.assertEqual(-0.1434, n_root.evaluate(data1))
        self.assertEqual(0.2, n_root.evaluate(data2))


if __name__ == '__main__':
    unittest.main()