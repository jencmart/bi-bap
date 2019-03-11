from .abstract_lts_regressor_fs import AbstractRegression
from .abstract_lts_regressor_fs import validate
import numpy as np
import math
import time


class FeasibleSolutionRegression(AbstractRegression):
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

    # ###########################################################################
    # ############### FIT #######################################################
    # ###########################################################################
    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 10,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True):

        # Init some properties
        X, y = validate(X, y, h_size, use_intercept)

        # concatenate to matrix
        if type(X) is not np.matrix:
            X = np.asmatrix(X)
        if type(y) is not np.matrix:
            y = np.asmatrix(y)
        self._data = np.asmatrix(np.concatenate([y, X], axis=1))

        self._p = self._data.shape[1] - 1
        self._N = self._data.shape[0]

        if h_size == 'default':
            self._h_size = math.ceil((self._N + self._p + 1) / 2)  # todo with or without intercept?
        else:
            self._h_size = h_size

        self.x_all = self._data[:, 1:]
        self.y_all = self._data[:, [0]]

        results = []

        time1 = time.process_time()
        # for all initial starts
        for i in range(num_starts):
            # generate random subset J, |J| = h and its complement M
            idx_initial, idx_rest = self.select_initial_h1()
            # save splitted data
            J = np.matrix(self._data[idx_initial], copy=True)
            M = np.matrix(self._data[idx_rest], copy=True)
            # do the refinement process
            res = self.refinement_process(J, M, idx_initial, idx_rest)

            # store the results
            results.append(res)

        # save the time
        self.time1_ = time.process_time() - time1
        self.time_total_ = self.time1_

        # select best results
        best_result = results[0]
        for res in results:
            if res.rss < best_result.rss:
                best_result = res

        # ... Store results
        theta_final = best_result.theta_hat

        if use_intercept:
            self.intercept_ = theta_final[-1, 0]  # last row last col
            self.coef_ = theta_final[:-1, 0]  # for all but last row,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = theta_final[:, 0]  # all rows, only first col

        self.h_subset_ = best_result.h_index
        self.rss_ = best_result.rss
        self.n_iter_ = best_result.steps

        self.coef_ = np.ravel(self.coef_)  # RAVELED

    # ###########################################################################
    # ############### ALL PAIRS  ################################################
    # ###########################################################################
    def go_through_all_pairs(self, J, M, inversion, residuals_J, residuals_M):
        delta = 1
        i_to_swap = None
        j_to_swap = None

        # go through all combinations
        for i in range(J.shape[0]):
            for j in range(M.shape[0]):
                # . calculate deltaRSS
                tmp_delta = self.calculate_delta_rss(J, M, inversion, residuals_J, residuals_M, i, j)
                # if delta rss < bestDeltaRss
                if tmp_delta < 0 and tmp_delta < delta:
                    delta = tmp_delta
                    i_to_swap = i
                    j_to_swap = j

        return i_to_swap, j_to_swap, delta

    # ###########################################################################
    # ############### REFINEMENT ################################################
    # ###########################################################################

    def refinement_process(self, J, M, idx_initial, idx_rest):
        steps = 0

        while True:
            # data for delata eqation
            y = J[:, [0]]
            x = J[:, 1:]
            inversion = (x.T * x).I
            theta = inversion * x.T * y  # OLS
            residuals_J = y - x * theta
            residuals_M = (M[:, [0]]) - (M[:, 1:]) * theta

            i_to_swap, j_to_swap, delta = self.go_through_all_pairs(J, M, inversion, residuals_J, residuals_M)
            if delta >= 0:
                break
            else:  # swap i and j [TOGHETHER WITH INDEXES] ; je to ok - SWAPUJEME SPRAVNE
                tmp = np.copy(J[i_to_swap])
                J[i_to_swap] = np.copy(M[j_to_swap])
                M[j_to_swap] = np.copy(tmp)
                idx_initial[i_to_swap], idx_rest[j_to_swap] = idx_rest[j_to_swap], idx_initial[i_to_swap]
                steps += 1

        # Save converged result
        # 1. calculate rs
        y_fin = J[:, [0]]
        x_fin = J[:, 1:]
        rss = (y_fin - x_fin * theta).T * (y_fin - x_fin * theta)
        rss = rss[0, 0]
        # 2. return in
        return self.Result(theta, idx_initial, rss, steps)

    # ###########################################################################
    # ############### DELTA RSS #################################################
    # ###########################################################################
    def calculate_delta_rss(self, J, M, inversion,
                            residuals_J, residuals_M, i, j):
        eiJ = residuals_J[i, 0]
        ejJ = residuals_M[j, 0]

        hii = J[i, 1:] * inversion * (J[i, 1:]).T  # 1xp * pxp * pX1
        hij = J[i, 1:] * inversion * (M[j, 1:]).T
        hjj = M[j, 1:] * inversion * (M[j, 1:]).T
        hii = hii[0, 0]
        hij = hij[0, 0]
        hjj = hjj[0, 0]

        nom = (ejJ * ejJ * (1 - hii)) - (eiJ * eiJ * (1 + hjj)) + 2 * eiJ * ejJ * hij
        denom = (1 - hii) * (1 + hjj) + hij * hij
        return nom / denom

    # ###########################################################################
    # ############### INITIAL H1 ################################################
    # ###########################################################################
    def select_initial_h1(self):
        # create random permutation
        idx_all = np.random.permutation(self._N)
        # cut first h indexes and save the rest
        idx_initial = idx_all[:self._h_size]
        idx_rest = idx_all[self._h_size:]

        return idx_initial, idx_rest
