from .abstract_lts_regressor_fs import AbstractRegression
from .abstract_lts_regressor_fs import validate
import cppimport.import_hook
import numpy as np
import math
import src.cpp.feasible_solution as lts
# lts = cppimport.imp("src/cpp/feasible_solution")
"""
cppimport
    For cppimport adhere this naming convention:
    cpp file: xxx.cpp
        inside: PYBIND11_MODULE(xxx, m)
    inside python module: my_import =  cppimport.imp("xxx")
"""


class FeasibleSolutionRegressionCPP(AbstractRegression):
    def __init__(self):
        super().__init__()
        self._data = None
        self._p = None
        self._N = None
        self._h_size = None
        # public
        self.n_iter_ = None
        self.coef_ = None
        self.intercept_ = None
        self.h_subset_ = None
        self.rss_ = None
        self.time1_ = None
        self.time_total_ = None

    # ############### FIT #######################################################
    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 10,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True):

        X, y = validate(X, y, h_size, use_intercept)

        # todo h_size including intercept?
        h_size = math.ceil((X.shape[0] + X.shape[1] + 1) / 2) if h_size == 'default' else h_size  # N + p + 1

        result = lts.fs_lts(X, y, num_starts, h_size)

        # Store result - weights first
        weights = result.get_theta()
        if use_intercept:
            self.intercept_ = weights[-1, 0]  # last row first col
            self.coef_ = np.ravel(weights[:-1, 0])  # for all but last column,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = np.ravel(weights[:, 0])  # all rows, only first col

        # Store rest of the attributes
        self.h_subset_ = result.get_h_subset()
        self.rss_ = result.get_rss()
        self.n_iter_ = result.get_n_inter()
        self.time1_ = result.get_time_1()
        self.time_total_ = self.time1_
