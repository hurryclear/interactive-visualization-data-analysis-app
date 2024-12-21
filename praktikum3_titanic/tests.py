import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from aufgabe3 import cross_validation_data

class TestCrossValidationData(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_cross_validation_data(self):
        data_train, data_test = cross_validation_data(self.X, self.y)
        
        # Check if the number of splits is correct
        self.assertEqual(len(data_train), 10)
        self.assertEqual(len(data_test), 10)
        
        # Check if each split has the correct shape
        for (X_train, y_train), (X_test, y_test) in zip(data_train, data_test):
            self.assertEqual(X_train.shape[1], self.X.shape[1])
            self.assertEqual(X_test.shape[1], self.X.shape[1])
            self.assertEqual(len(y_train) + len(y_test), len(self.y))
        
        # Check if the splits are stratified
        for (X_train, y_train), (X_test, y_test) in zip(data_train, data_test):
            self.assertEqual(np.sum(y_train == 0), np.sum(y_train == 1))
            self.assertEqual(np.sum(y_test == 0), np.sum(y_test == 1))

if __name__ == '__main__':
    unittest.main()