from abc import ABC, abstractmethod
import numpy as np

# currently support for np.ndarray and matrix
def validate(X, y, h_size, use_intercept):
    if X is None or not isinstance(X, (np.ndarray, np.matrix)):
        raise Exception('X must be  type array or np.ndarray or np.matrix')
    if y is None or not isinstance(y, (np.ndarray, np.matrix)):
        raise Exception('y must be  type array or np.ndarray or np.matrix')

    if X.ndim == 1:
        X = np.reshape(X, [X.shape[0], 1])
    if y.ndim == 1:
        y = np.reshape(y, [y.shape[0], 1])

    if type(X) is not np.ndarray:
        X = np.ndarray(X)
    if type(y) is not np.ndarray:
        y = np.ndarray(y)

    if y.ndim != 1:
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError('y must be 1D array')
    if y.shape[0] != X.shape[0]:
        raise ValueError('X and y must have same number of samples')

    if X.shape[0] < 1:  # expects N >= 1
        raise ValueError('You must provide at least one sample')

    if X.ndim < 1:  # expects p >=1
        raise ValueError('X has zero dimensions')

    if h_size != 'default':
        if h_size > X.shape[0]:
            raise ValueError('H_size must not be > number of samples')
        if h_size < 1:
            raise ValueError('H_size must be > 0 ; preferably (n + p + 1) / 2 <= h_size <= n ')

    if use_intercept:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    return X, y


class AbstractRegression(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 10,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True):
        raise NotImplementedError("You must implement this")

    # for storage only
    class Result:
        def __init__(self, theta_hat, h_index, rss, steps):
            self.theta_hat = theta_hat
            self.h_index = h_index
            self.rss = rss
            self.steps = steps
