import cppimport
import numpy as np
import math

###########################
# To work way you expect, you must adhere the naming convention right:
# xxx.cpp
#	PYBIND11_MODULE(xxx, m)

# something.py
#	my_import =  cppimport.imp("xxx")
###########################

eigen_lts = cppimport.imp("../src/feasible_solution")


class FastLtsEigenRegressor:
    def __init__(self):
        # public
        self.n_iter_ = None
        self.coef_ = None
        self.intercept_ = None
        self.h_subset_ = None
        self.rss_ = None
        self.time1_ = None
        self.time2_ = None
        self.time3_ = None
        self.time_total_ = None

    # currently support for np.ndarray and matrix
    def _validate(self, X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold,
                  use_intercept):
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

        if max_c_steps < 1:  # expects max_steps => 1
            raise ValueError('max_c_steps must be >= 1')

        if num_start_c_steps < 1:  # expects num_start_steps => 1
            raise ValueError('num_start_c_steps must be >= 1')

        if num_starts_to_finish < 1:  # expects num_starts_to_finish >= 1
            raise ValueError('num_starts_to_finish must be >= 1')

        if threshold < 0:  # expects threshold >= 0
            raise ValueError('threshold must be >= 0')

        if use_intercept:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        return X, y


    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 500,
            num_start_c_steps: 'number of initial C steps' = 2,
            num_starts_to_finish: 'number of H3 which`ll to finish' = 10,
            max_c_steps: 'self explanatory' = 50,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True,
            threshold: 'stopping criterion Qold Qnew' = 1e-6):


        X, y = self._validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold,
                              use_intercept)

        # todo - include intercept or not? now - p include intercept..
        _h_size = math.ceil((X.shape[0] + X.shape[1] +1) / 2) if h_size == 'default' else h_size  # N + p + 1

        eigen_result = eigen_lts.fast_lts(X, y, num_starts, num_start_c_steps, num_starts_to_finish, _h_size,
                                          max_c_steps, threshold)

        # ... Store best result
        weights = eigen_result.get_theta()
        if use_intercept:
            self.intercept_ = weights[-1, 0]  # last row first col
            self.coef_ = np.ravel(weights[:-1, 0])  # for all but last column,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = np.ravel(weights[:, 0])  # all rows, only first col

        self.h_subset_ = eigen_result.get_h_subset()
        self.rss_ = eigen_result.get_rss()
        self.n_iter_ = eigen_result.get_n_inter()
        self.time1_ = eigen_result.get_time_1()
        self.time2_ = eigen_result.get_time_2()
        self.time3_ = eigen_result.get_time_3()
        self.time_total_ = self.time1_ + self.time2_ + self.time3_

