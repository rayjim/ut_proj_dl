'''
Created on Nov 19, 2016

@author: raybao
'''
import unittest
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import ml_apps.mlp_tf as mlp


def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'),
                               random_state=42)

    mnist_X = mnist_X / 255.0

    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                        test_size=0.2,
                                                        random_state=42)

    return (train_X, test_X, train_y, test_y)


class TestMLP(unittest.TestCase):

    def setUp(self):
        self.train_X, self.test_X, self.train_y, self.test_y = load_mnist()

    def test_HomeWorkMLP(self):
        # validate for small dataset
        train_X_mini = self.train_X[:1000]
        train_y_mini = self.train_y[:1000]
        test_X_mini = self.test_X[:1000]
        test_y_mini = self.test_y[:1000]
        pred_y = mlp.homework(train_X_mini, train_y_mini, test_X_mini)
        score = f1_score(test_y_mini, pred_y)
        self.assertGreater(score, 0.85, "The f1 score is %f: too low" % score)


if __name__ == "__main__":  # pragma: nocoverage
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
