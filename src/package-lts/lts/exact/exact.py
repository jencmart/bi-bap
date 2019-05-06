from lts.exact.helpers import AbstractRegression
from lts.exact.helpers import validate
import numpy as np
import math
import time
from itertools import combinations
from itertools import product

import cppimport.import_hook
import lts.exact.cpp.exact as cpp_solution

from scipy import linalg

"""
# lts = cppimport.imp("feasible/cpp/feasible_solution")
cppimport
    For cppimport adhere this naming convention:
    cpp file: xxx.cpp
        inside: PYBIND11_MODULE(xxx, m)
    inside python module: my_import =  cppimport.imp("xxx")
"""


class LTSRegressorExactCPP(AbstractRegression):

    def __init__(self,
                 use_intercept=True,
                 algorithm: 'str, ‘exa’, ‘bab’ or ‘bsa’, default: ‘bab’' = 'bab',
                 calculation: 'str, ‘inv’, ‘qr’, default: ‘qr’' = 'inv'):
        super().__init__()

        # set using intercept
        self._use_intercept = use_intercept

        # algorithm and calculation
        self._alg = algorithm
        self._calculation = calculation

        self._h_size = 0

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
            h_size: 'int, default:(n + p + 1) / 2' = 'default',
            set_rss: 'smallest rss known on some h_subset, improves bsa performance' = None):

        # Init some properties
        X, y = validate(X, y, h_size, self._use_intercept)

        p = X.shape[1]
        n = X.shape[0]
        h_size = calculate_h_size(n, p, h_size)

        if self._alg == 'exa':
            int_alg = 0
        elif self._alg == 'bab':
            int_alg = 1
        elif self._alg == 'bsa':
            int_alg = 2
        else:
            raise ValueError('param. algorithm must be one fo the strings: ‘exa’, ‘bab’ or ‘bsa’')

        if self._calculation == 'inv':
            int_calc = 0
        elif self._calculation == 'qr':
            int_calc = 1
        else:
            raise ValueError('param. calculation must be one fo the strings: ‘inv’ or ‘qr’')

        if set_rss is None:
            set_rss = -1
        else:
            set_rss += 0.0001

        result = cpp_solution.exact_lts(X, y, h_size, int_alg, int_calc, set_rss)

        # todo - recalculate theta

        # Store result - weights first
        weights = result.get_theta()
        if self._use_intercept:
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


def calculate_h_size(n, p, h_size):
    if h_size == 'default':
        s = math.floor((n + p + 1) / 2)  # greatest integer function ~ floor
    else:
        s = h_size

    return s


class LTSRegressorExact(AbstractRegression):

    def __init__(self,
                 use_intercept=True,
                 algorithm: 'str, ‘fsa’ or ‘mmea’, default: ‘fsa’' = 'fsa',
                 calculation: 'str, ‘inv’, ‘qr’, default: ‘qr’' = 'inv'):
        super().__init__()

        # set using intercept
        self._use_intercept = use_intercept

        # algorithm and calculation
        self._alg = algorithm
        self._calculation = calculation

        self._h_size = 0

        # public
        self.n_iter_ = None
        self.coef_ = None
        self.intercept_ = None
        self.h_subset_ = None
        self.rss_ = None
        self.time1_ = None
        self.time_total_ = None

    def fit(self, X, y,
                  h_size: 'default := (n + p + 1) / 2' = 'default',
                  set_rss: 'smallest rss known on some h_subset, improves bsa performance' = None):

        # Init some properties
        X, y = validate(X, y, h_size, self._use_intercept)

        # concatenate to matrix
        if type(X) is not np.matrix:
            X = np.asmatrix(X)
        if type(y) is not np.matrix:
            y = np.asmatrix(y)
        data = np.asmatrix(np.concatenate([y, X], axis=1))

        # h size
        p = data.shape[1] - 1
        n = data.shape[0]
        self._h_size = calculate_h_size(n, p, h_size)

        if h_size == 0 or h_size == 1:
            print('h_size must be at least 2')
            exit(1)

        self.J = np.matrix(data, copy=True)

        results = []

        # start measuring the time
        time1 = time.process_time()

        if self._alg == 'exa':
            # do the refinement process
            res = self.refinement_exhaustive()

        elif self._alg == 'bab':
            # do the refinement process
            res = self.refinement_bab_lts()
        else:
            if set_rss is None:
                res = self.refinement_bsa()
            else:
                res = self.refinement_bsa(set_rss + 0.00001)

        # store the result
        results.append(res)

        # stop measuring the time
        self.time1_ = time.process_time() - time1
        self.time_total_ = self.time1_

        self.save_the_best(results)

    # select and store the best solution ...
    def save_the_best(self, results):
        # select best results
        best_result = results[0]
        for res in results:
            if res.rss < best_result.rss:
                best_result = res

        # ... Store results
        theta_final = best_result.theta_hat

        if self._use_intercept:
            self.intercept_ = theta_final[-1, 0]  # last row last col
            self.coef_ = theta_final[:-1, 0]  # for all but last row,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = theta_final[:, 0]  # all rows, only first col

        self.h_subset_ = best_result.h_index
        self.rss_ = best_result.rss
        self.n_iter_ = best_result.steps

        self.coef_ = np.ravel(self.coef_)  # RAVELED

    # ################################################################################
    # ############ E X H A U S T I V E ###############################################
    # ################################################################################
    def refinement_exhaustive(self):
        rss_min = float('inf')
        indexes_min = None
        theta_min = None

        all_indexes = np.arange(self.J.shape[0])

        # Get all combinations of all_indexes
        # and length h_size
        all_comb = combinations(all_indexes, self._h_size)

        # for each combination
        for comb in list(all_comb):
            indexes = list(comb)
            data = self.J[indexes]
            theta, inversion = self.theta_inv(data)
            rss = self.rss(data, theta)

            if rss < rss_min:
                rss_min = rss
                theta_min = theta
                indexes_min = indexes

        steps = 0
        indexes_min = np.asarray(indexes_min)
        return self.Result(theta_min, indexes_min, rss_min, steps)

    # ################################################################################
    # ############ E X A C T  -  B A B ###############################################
    # ################################################################################
    def refinement_bab_lts(self):
        self.bab_rss_min = float('inf')
        self.bab_indexes = None
        self.cuts = 0

        a = []
        b = list(range(self.J.shape[0]))
        self.traverse_recursive(a, b, 0, None, None, None)

        bab_theta, _, _, _ = self.theta_qr(self.J[self.bab_indexes])
        return self.Result(bab_theta, self.bab_indexes, self.bab_rss_min, self.cuts)

    def traverse_recursive(self, a, b, depth, rss, theta, inversion):
        # bottom of the tree
        if depth == self._h_size:
            # calculate gama plus and new RSS
            gamma_plus = self.gamma_plus_inv(self.J, a[-1], inversion, theta)
            rss_here = rss + gamma_plus

            # possibly update result
            if rss_here < self.bab_rss_min:
                self.bab_rss_min = rss_here
                self.bab_indexes = np.copy(a)

                # update theta and inversion - calculate at the end ...
                # self.bab_theta, _ = self.theta_plus_inversion_plus(theta, inversion, self.J, a[-1])
            return

        # leaf, but cannot go deeper
        if len(b) == 0:
            exit(3)

        # before 'root' - we need at least dept p because we assume regularity
        if len(a) < self.J.shape[1]:  # nothing in root
            rss_here = rss
            theta_here = theta
            inversion_here = inversion

        # 'root' - calculate for the first time at the dept p
        elif len(a) == self.J.shape[1]:
            theta_here, inversion_here = self.theta_inv(self.J[a])
            rss_here = self.rss(self.J[a], theta_here)

        # ordinary edge
        else:
            # update RSS
            gamma_plus = self.gamma_plus_inv(self.J, a[-1], inversion, theta)
            rss_here = rss + gamma_plus

            # check bounding condition and eventually cut all the branches
            if rss_here >= self.bab_rss_min:
                self.cuts += 1
                return

            # update theta and inversion
            theta_here, inversion_here = self.theta_plus_inversion_plus(theta, inversion, self.J, a[-1])

        # traverse tree deeper
        aa = a.copy()
        bb = b.copy()

        # sort first --- we use set A to calculate h-subset
        # lets calculate all partial increments for every element in B subset
        # >>> do not leads to the speedup <<<
        # if len(a) >= self.J.shape[1]:
        #     gamma_list = []
        #     for j in bb:
        #         # calculate rss change after including row j
        #         gamma_plus = self.gamma_plus_inv(self.J, j, inversion_here, theta_here)
        #         gamma_list.append(gamma_plus)
        #
        #     # sort from lowest to highest (we traverse tree in LTR order)
        #     bb = [elem for _, elem in sorted(zip(gamma_list, bb))]

        while len(bb) > 0:
            if len(aa) + len(bb) < self._h_size:  # not enough to produce h subset in ancestors
                break

            # move index from from A to B
            aa.append(bb[0])
            del bb[0]

            # go deeper
            self.traverse_recursive(aa, bb, depth + 1, rss_here, theta_here, inversion_here)

            # remove index from the end of the A
            del aa[-1]

        # return up from the current node
        return

    # ################################################################################
    # ############ E X A C T  -  B S A ###############################################
    # ################################################################################
    def refinement_bsa(self, rss=None):
        if rss is None:
            rss_min = float('inf')
        else:
            rss_min = rss

        theta_min = None
        h_subset_min = None
        p = self.J.shape[1] - 1
        cuts = 0

        # Get all combinations of all_indexes
        # and length p + 1
        all_indexes = np.arange(self.J.shape[0])
        all_comb = combinations(all_indexes, p + 1)

        for comb in list(all_comb):  # for all p+1 indexes ... \binom{n}{p+1}
            indexes = list(comb)

            # store first index
            first = indexes[0]
            del indexes[0]

            x1 = self.J[first, 1:]   # 1 x p
            y1 = self.J[first, [0]]  # 1 x 1

            x_rest = np.copy(self.J[indexes, 1:])  # p x p
            y_rest = np.copy(self.J[indexes, 0])   # p x 1

            # for all 2^p signs {+,-}
            all_sign_permut = list(map(list, product([0, 1], repeat=p)))  # (000)(001)(010)(011)(100)(101)(110)(111)
            for signs in all_sign_permut:

                # create equations
                for i in range(p):
                    if signs[i] == 1:
                        x_rest[i] = x_rest[i] - x1
                        y_rest[i] = y_rest[i] - y1
                    else:
                        x_rest[i] = x_rest[i] + x1
                        y_rest[i] = y_rest[i] + y1

                # solve theta
                data = np.asmatrix(np.concatenate([y_rest, x_rest], axis=1))
                theta_curr, q, r, r1 = self.theta_qr(data)

                # check if solution is unique
                if np.count_nonzero(r[-1, :]) == 0:  # if last row is zero ... not regular
                    continue

                #  calculate all residuals, square them and sort them
                all_residuals = self.J[:, [0]] - self.J[:, 1:] * theta_curr
                all_residuals = np.square(all_residuals)
                all_residuals = np.ravel(all_residuals)
                sort_args = np.argsort(all_residuals)  # only the argsort

                # calculate xi1 residuum
                x1_res = ((y1 - np.dot(x1, theta_curr))[0, 0])**2

                # if r_{x1} == r_h
                res_h = all_residuals[sort_args[self._h_size - 1]]  # r_h residuum
                res_h_1 = all_residuals[sort_args[self._h_size]]    # r_{h+1} residuum

                if math.isclose(x1_res, res_h, rel_tol=1e-9):

                    # if r_h == r_{h+1} .. find all corresponding h subsets; #subsets ~ binom{p}{l+1}; max binom{p}{p/2}
                    if math.isclose(res_h, res_h_1, rel_tol=1e-9):
                        p = self.J.shape[1] - 1
                        all_h_subsets = self.all_h_subsets_bsa(all_residuals, sort_args, p, rss_min)

                        if all_h_subsets is None:  # BSA-BAB speedup
                            cuts += 1
                            continue

                    # else.. corresponding h subset is unique
                    else:
                        all_h_subsets = [sort_args[:self._h_size]]

                    # for each h_subset in relation with theta calculate OLS estimate
                    for h_sub_indexes in all_h_subsets:
                        theta, _, _, _ = self.theta_qr(self.J[h_sub_indexes])
                        rss = self.rss(self.J[h_sub_indexes], theta)

                        # if current new minimum .. store it
                        if rss < rss_min:
                            rss_min = rss
                            theta_min = theta
                            h_subset_min = h_sub_indexes

        return self.Result(theta_min, h_subset_min, rss_min, cuts)

    def all_h_subsets_bsa(self, residuals, sort_args, p, rss_min):

        res_h = residuals[sort_args[self._h_size - 1]]  # r_h residuum

        # find smallest index i; i <= h;  so that r_i = r_h
        idx_i = self._h_size-1
        for i in reversed(range(self._h_size-1)):  # r_h, r_{h-1}, ... ,r_{0}
            if math.isclose(residuals[sort_args[i]], res_h, rel_tol=1e-9):  # e-9 is default btw..
                idx_i = i
            else:
                break

        # find greatest index j; j >= h+1 ;  so that r_j = r_{h+1} ; [ h+1 because we know that r_h == r_{h+1} ]
        idx_j = self._h_size
        for j in range(self._h_size, residuals.shape[0]):
            if math.isclose(residuals[sort_args[j]], res_h, rel_tol=1e-9):
                idx_j = j
            else:
                break

        # BSA-BAB speedup
        cnt_unique = idx_i  # idx_1 - 1 + 1 (- because at idx_i is smallest r_i = r_h) (+ because it is index)
        if cnt_unique >= p:
            begin = sort_args[:idx_i]
            theta, _, _, _ = self.theta_qr(self.J[begin])
            rss = self.rss(self.J[begin], theta)
            if rss >= rss_min:
                return None

        # create list from those indexes [i, i+1, ... , j]
        idx_list = list(range(idx_i, idx_j+1))

        # create all combinations of size:= self._h_size-1 - idx_i + 1
        # #equal residuals from i to h included (h-i+1)
        # we are using indexes so (h-1-i+1) ... h - idx_i where h is #resuidals
        comb = combinations(idx_list, self._h_size - idx_i)

        # store all corresponding subsets
        list_of_subsets = []

        # start with first i unique indexes [0, 1, ... i-1]
        begin = sort_args[:idx_i]

        # and append rest of the indexes for each combination
        for appendix in list(comb):
            appendix = sort_args[list(appendix)]
            concatenated_list = np.concatenate((begin, appendix), axis=0)
            list_of_subsets.append(concatenated_list)

        return list_of_subsets

    # #####################################################################################
    # ################# HELPERS ###########################################################
    # #####################################################################################
    # calculate inversion and theta (OLS estimate)
    @staticmethod
    def theta_inv(J):
        y = J[:, [0]]
        x = J[:, 1:]

        inversion = (x.T * x).I
        theta = inversion * x.T * y  # OLS
        return theta, inversion

    # calculate sum of squared residuals O(np)
    @staticmethod
    def rss(J, theta):
        y_fin = J[:, [0]]
        x_fin = J[:, 1:]
        residuals = y_fin - x_fin * theta
        rss = residuals.T * residuals
        rss = rss[0, 0]
        return rss

    # Calculate theta using normal equation: R1 theta = Q1y
    def theta_qr(self, J):
        # Q ... n x n
        # R ... n x p
        q, r = linalg.qr(J[:, 1:])  # X = QR ; x.T x = R.T R ;

        # Q.T *  ( x * w - y ) ^ 2
        # Q.T * Q * R * w - Q.T * y
        # R * w - Q.T * y
        # R * w = Q.T * y
        theta, r1 = self.theta_from_qr(q, r, J)

        return theta, q, r, r1

        # Update theta using normal equation: R1 theta = Q1y
        # calculate theta from QR decomposition O(p^2)

    @staticmethod
    def theta_from_qr(q, r, J):
        y = J[:, [0]]
        p = r.shape[1]
        # r1 p x p
        r1 = r[:p, :]  # only first p rows
        qt = q.T
        q1 = qt[:p, :]  # only first p rows

        # solve the equation x w = c for x, assuming a is a triangular matrix
        theta = linalg.solve_triangular(r1, q1 * y)  # p x substitution
        return theta, r1

        # calculate gama plus  O(p^2)

    def gamma_plus_inv(self, M, j, inversion, theta):
        i_m_i = self.dot_idx_m_idx(M, j, inversion)
        xi = M[j, 1:]
        yi = M[j, [0]]  # 1 x 1
        yi_xi_theta = (yi - np.dot(xi, theta))[0, 0]
        gamma_plus = (yi_xi_theta ** 2) / (1 + i_m_i)
        return gamma_plus

    @staticmethod
    def dot_idx_m_idx(A, idx, inversion):
        x_idx = A[idx, 1:]
        idx_m_idx = np.dot(np.dot(x_idx, inversion), x_idx.T)
        idx_m_idx = idx_m_idx[0, 0]
        return idx_m_idx

    # update theta and inversion when row is included O(p^2)
    @staticmethod
    def theta_plus_inversion_plus(theta, inversion, M, j_swap):
        i_m_i = LTSRegressorExact.dot_idx_m_idx(M, j_swap, inversion)  # O(p^2)
        # Theta plus
        xi = M[j_swap, 1:]  # 1 x p
        yi = M[j_swap, [0]]  # 1 x 1
        w = -1 / (1 + i_m_i)  # 1x1
        u = np.dot(inversion, xi.T)  # p x 1  # O(p^2)
        theta_plus = theta + (-1 * (yi - np.dot(xi, theta))[0, 0] * (w * u))  # O(p)  # !!!! (changed [* -1] )

        # Inversion plus
        inversion_plus = inversion + w * np.dot(u, u.T)  # O(p^2)

        return theta_plus, inversion_plus

    # ############################################################################
    # ############################################################################
    # ######## USABLE FOR BAB QR #################################################
    # ############################################################################
    # ############################################################################
    # ############################################################################

    # # Calculate theta directly from ~M without c = Q1y
    #  def calculate_theta_fii(self, Ja):
    #      J = np.copy(Ja)
    #      J = np.asmatrix(J)
    #
    #      # move first col (y) to last col so we'll have (X|y)
    #      first_col = J[:, [0]]
    #      J[:, : -1] = J[:, 1:]
    #      J[:, [-1]] = first_col
    #
    #      qM, rM = linalg.qr(J)
    #      theta, rss, r1 = self.update_theta_fii(rM)
    #
    #      return theta, qM, rM, r1, rss
    #
    #  # Update theta directly from ~M without c = Q1y
    #  def update_theta_fii(self, rM):
    #      p = rM.shape[1] - 1
    #      r1 = rM[:p, : -1]
    #      fii = rM[:p, [-1]]
    #      theta = linalg.solve_triangular(r1, fii)
    #      rss = rM[p, p] ** 2
    #
    #      return theta, rss, r1
    #
    #  def refinement_process_fs_moe_qr_extended(self, J, M, idx_initial, idx_rest):
    #      steps = 0
    #
    #      # Calculate QR decompositon along with theta and RSS directly from (X|y)
    #      theta, qM, rM, r1, rss = self.calculate_theta_fii(J)
    #
    #      # Calculate residuals e
    #      residuals_J, residuals_M = self.all_residuals(J, M, theta)
    #
    #      while True:
    #          i_to_swap, j_to_swap, delta = self.all_pairs_fsa_oe_qr(J, M, r1, rss, residuals_J, residuals_M)
    #
    #          if delta >= 1:
    #              break
    #
    #          else:
    #              # Save row to swap in QR
    #              row_to_addM = np.copy(M[j_to_swap, :])
    #              # move first elem (y) to last col
    #              first_col = row_to_addM[:, [0]]
    #              row_to_addM[:, : -1] = row_to_addM[:, 1:]
    #              row_to_addM[:, [-1]] = first_col
    #
    #              # Update J and M arrays and also idx array by means of swapped rows
    #              self.swap_row_J_M(J, M, idx_initial, idx_rest, i_to_swap, j_to_swap)
    #
    #              # Update QR
    #              qM, rM = self.qr_insert(qM, rM, row_to_addM, i_to_swap + 1)
    #              qM, rM = self.qr_delete(qM, rM, i_to_swap)
    #
    #              # Update theta, R1, RSS
    #              theta, rss, r1 = self.update_theta_fii(rM)
    #
    #              # calculate residuals M and J
    #              residuals_J, residuals_M = self.all_residuals(J, M, theta)
    #
    #              steps += 1
    #
    #      return self.Result(theta, idx_initial, rss, steps)
