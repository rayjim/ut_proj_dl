import numpy as np


def homework(train_X, train_y, test_X, k=3):
    """
       homework for training data
       train_X: feature vecotr for the training
       train_Y: label for training data
       test_X: feature vecotrs for testing
       k: temporary decided
    """
    pred_y = []
    batch_size = 2000
    train_X_norm = np.linalg.norm(train_X, axis=1)
    train_X = (train_X/train_X_norm[:, None])
    test_X_norm = -np.linalg.norm(test_X, axis=1)
    test_X = (test_X/test_X_norm[:, None])
    train_X = train_X.T

    def find_k(temp):
        top_k = temp[:k]
        a = train_y[top_k]
        i_result = np.bincount(a).argmax()
        return i_result
    for i_test in range(0, test_X.shape[0], batch_size):
        end = min(i_test+batch_size, test_X.shape[0])
        dotted_result = np.dot(test_X[i_test:end], train_X)
        temp = np.argpartition(dotted_result, k, axis=-1)
        pred_y_batch = np.apply_along_axis(find_k, axis=1, arr=temp)
        pred_y.extend(pred_y_batch)
    return pred_y


if __name__ == "__main__":  # pragma: no coverage
    pass
