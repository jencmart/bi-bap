from lts.feasible.helpers import AbstractRegression
from lts.feasible.helpers import validate
import numpy as np
import math
import time
import cppimport.import_hook
import lts.feasible.cpp.feasible_solution as cpp_solution
from scipy import linalg

"""
# lts = cppimport.imp("feasible/cpp/feasible_solution")
cppimport
    For cppimport adhere this naming convention:
    cpp file: xxx.cpp
        inside: PYBIND11_MODULE(xxx, m)
    inside python module: my_import =  cppimport.imp("xxx")
"""


class FSRegressorCPP(AbstractRegression):

    def __init__(self):
        super().__init__()
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
            num_starts: 'number of starting subsets' = 10,
            custom_start_subset: 'array of size h, the custom starting subset' = None,
            max_steps: 'max number of steps to converge' = 50,
            h_size: 'int, default:(n + p + 1) / 2' = 'default',
            use_intercept=True,
            algorithm: 'str, ‘fsa’, ‘oea’, ‘moea’ or ‘mmea’, default: ‘fsa’' = 'default',
            calculation: 'str, ‘inv’, ‘qr’, default: ‘qr’' = 'inv'):

        X, y = validate(X, y, h_size, use_intercept)

        # todo h_size including intercept?
        h_size = math.ceil((X.shape[0] + X.shape[1] + 1) / 2) if h_size == 'default' else h_size  # N + p + 1

        result = cpp_solution.fs_lts(X, y, num_starts, h_size)

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


class FSRegressor(AbstractRegression):
    def __init__(self,
                 num_starts: 'number of starting subsets' = 10,
                 max_steps: 'max number of steps to converge' = 50,
                 use_intercept=True,
                 algorithm: 'str, ‘fsa’, ‘oea’, ‘moea’ or ‘mmea’, default: ‘fsa’' = 'fsa',
                 calculation: 'str, ‘inv’, ‘qr’, default: ‘qr’' = 'inv'):
        super().__init__()

        # number of initial starts starts
        self._num_starts = num_starts

        # maximum number of iterations
        self._max_steps = max_steps

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

    def fit(self, X, y, h_size: 'int, default:(n + p + 1) / 2' = 'default', index_subset=None):

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
        if h_size == 'default':
            self._h_size = math.ceil((n + p + 1) / 2)  # todo with or without intercept?
        else:
            self._h_size = h_size

        results = []

        # start measuring the time
        time1 = time.process_time()

        if index_subset is None:

            # generate T random starts
            for i in range(self._num_starts):

                # generate random subset Ones - Xm, |Xm| = h and its complement Zeroes
                idx_ones, idx_zeroes = self.generate_random_start(n)

                # create matrices based on those indexes
                J = np.matrix(data[idx_ones], copy=True)
                M = np.matrix(data[idx_zeroes], copy=True)

                # calculation using inversion
                if self._calculation == 'inv':
                    if self._alg == 'fsa':
                        # do the refinement process
                        res = self.refinement_process_fsa_inv(J, M, idx_ones, idx_zeroes)

                    # elif self._alg == 'oea':
                    #     # do the refinement process
                    #     res = self.refinement_process_oea_inv(J, M, idx_ones, idx_zeroes)

                    elif self._alg == 'moea':
                        # do the refinement process
                        res = self.refinement_process_moea_inv(J, M, idx_ones, idx_zeroes)

                    elif self._alg == 'mmea':
                        # do the refinement process
                        res = self.refinement_process_mmea_inv(J, M, idx_ones, idx_zeroes)
                    else:
                        raise ValueError('param. algorithm must be one fo the strings: ‘fsa’, ‘oea’, ‘moea’ or ‘mmea’')

                    # store the results
                    results.append(res)

                # calculation using qr decomposition
                elif self._calculation == 'qr':
                    if self._alg == 'fsa':
                        # do the refinement process
                        res = self.refinement_process_fsa_qr(J, M, idx_ones, idx_zeroes)

                    # elif self._alg == 'oea':
                    #     # do the refinement process
                    #     res = self.refinement_process_oea_qr(J, M, idx_ones, idx_zeroes)

                    elif self._alg == 'moea':
                        # do the refinement process
                        res = self.refinement_process_moea_qr(J, M, idx_ones, idx_zeroes)

                    elif self._alg == 'mmea':
                        # do the refinement process
                        res = self.refinement_process_mmea_qr(J, M, idx_ones, idx_zeroes)
                    else:
                        raise ValueError(
                            'param. algorithm must be one fo the strings: ‘fsa’, ‘oea’, ‘moea’ or ‘mmea’')

                    # store the results
                    results.append(res)

                else:
                    raise ValueError('param. calculation must be one fo the strings: ‘inv’ or ‘qr’')

        else:
            for subs in range(self._num_starts):
                # generate random subset J, |J| = h and its complement M
                idx_ones, idx_zeroes = self.generate_random_start(n)
                # save splitted data
                J = np.matrix(data[idx_ones], copy=True)
                M = np.matrix(data[idx_zeroes], copy=True)
                # do the refinement process
                res = self.refinement_process_fsa_inv(J, M, idx_ones, idx_zeroes)

                # store the results
                results.append(res)

        # stop measuring the time
        self.time1_ = time.process_time() - time1
        self.time_total_ = self.time1_

        self.save_the_best(results)

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

    # ###########################################################################
    # ############### ALL PAIRS  ################################################
    # ###########################################################################
    def all_pairs_fsa_inv(self, J, M, inversion, residuals_J, residuals_M):
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
    # ############### REFINEMENT FSA INV ########################################
    # ###########################################################################

    # calculate inversion and theta (OLS estimate)
    @staticmethod
    def theta_inv(J):
        y = J[:, [0]]
        x = J[:, 1:]

        inversion = (x.T * x).I
        theta = inversion * x.T * y  # OLS
        return theta, inversion

    # calculate r_1 ... r_n    O(np)
    @staticmethod
    def all_residuals(J, M, theta):
        residuals_J = J[:, [0]] - J[:, 1:] * theta
        residuals_M = (M[:, [0]]) - (M[:, 1:]) * theta

        return residuals_J, residuals_M

    @staticmethod
    def swap_xi_xj(J, M, idx_initial, idx_rest, i, j):
        tmp = np.copy(J[i])
        J[i] = np.copy(M[j])
        M[j] = np.copy(tmp)
        idx_initial[i], idx_rest[j] = idx_rest[j], idx_initial[i]

    # calculate sum of squared residuals O(np)
    @staticmethod
    def rss(J, theta):
        y_fin = J[:, [0]]
        x_fin = J[:, 1:]
        residuals = y_fin - x_fin * theta
        rss = residuals.T * residuals
        rss = rss[0, 0]
        return rss

    def refinement_process_fsa_inv(self, J, M, idx_ones, idx_zeroes):
        steps = 0

        while True:

            # calculate theta and inversion  # O(np^2)
            theta, inversion = self.theta_inv(J)

            # calculate residuals r_1 ... r_n  O(np)
            resid_J, resid_M = self.all_residuals(J, M, theta)

            # find the optimal swap   O(n^2p^2)
            i_swap, j_swap, delta = self.all_pairs_fsa_inv(J, M, inversion, resid_J, resid_M)

            # strong necessary condition satisfied
            if delta >= 0:
                break

            # swap observations
            else:
                self.swap_xi_xj(J, M, idx_ones, idx_zeroes, i_swap, j_swap)
                steps += 1

        # calculate rss and return result
        rss = (resid_J.T * resid_J)[0, 0]
        return self.Result(theta, idx_ones, rss, steps)

    # update theta and inversion when row is included O(p^2)
    @staticmethod
    def theta_plus_inversion_plus(theta, inversion, M, j_swap):
        i_m_i = FSRegressor.dot_idx_m_idx(M, j_swap, inversion)  # O(p^2)
        # Theta plus
        xi = M[j_swap, 1:]  # 1 x p
        yi = M[j_swap, [0]]  # 1 x 1
        w = -1 / (1 + i_m_i)  # 1x1
        u = np.dot(inversion, xi.T)  # p x 1  # O(p^2)
        theta_plus = theta + (-1 * (yi - np.dot(xi, theta))[0, 0] * (w * u))  # O(p)  # !!!! (changed [* -1] )

        # Inversion plus
        inversion_plus = inversion + w * np.dot(u, u.T)  # O(p^2)

        return theta_plus, inversion_plus

    # update theta and inversion when row is excluded O(p^2)
    @staticmethod
    def theta_minus_inversion_minus(theta_plus, inversion_plus, J, i_swap):
        j_m_j = FSRegressor.dot_idx_m_idx(J, i_swap, inversion_plus)
        # Theta plus minus
        xj = J[i_swap, 1:]
        yj = J[i_swap, [0]]
        wj = -1 / (1 - j_m_j)
        uj = np.dot(inversion_plus, xj.T)
        theta_plus_minus = theta_plus + ((yj - np.dot(xj, theta_plus))[0, 0] * (wj * uj))

        # Inversion plus minus
        inversion_plus_minus = inversion_plus - wj * np.dot(uj, uj.T)  # col * row = matrix

        return theta_plus_minus, inversion_plus_minus

    # ###########################################################################
    # ############### REFINEMENT MOEA INV #######################################
    # ###########################################################################
    def refinement_process_moea_inv(self, J, M, idx_ones, idx_zeroes):
        steps = 0

        # calculate theta and inversion  # O(np^2)
        theta, inversion = self.theta_inv(J)

        # calculate rss  O(np)
        rss = self.rss(J, theta)

        # shortcut
        if M.shape[0] == 0:
            return self.Result(theta, idx_ones, rss, steps)

        while True:

            # calculate residuals r_1 ... r_n    O(np)
            res_J, res_M = self.all_residuals(J, M, theta)

            # find the optimal swap - j add ; i remove O(n^2p^2)
            i_swap, j_swap, rho = self.all_pairs_moea_inv(J, M, inversion, rss, res_J, res_M)

            # strong necessary condition satisfied
            if rho >= 1:
                break

            # swap observations and update rss, theta and inversion
            else:

                # update rss
                rss = rss * rho

                # update theta -> theta_plus ; inversion -> inversion_plus  O(p^2)
                theta_plus, inversion_plus = self.theta_plus_inversion_plus(theta, inversion, M, j_swap)

                # update theta_plus -> theta_minus ; inversion_plus -> inversion_minus  O(p^2)
                theta, inversion = self.theta_minus_inversion_minus(theta_plus, inversion_plus, J, i_swap)

                # swap observations
                self.swap_xi_xj(J, M, idx_ones, idx_zeroes, i_swap, j_swap)
                steps += 1

        return self.Result(theta, idx_ones, rss, steps)

    # ###########################################################################
    # ############### REFINEMENT MOEA QR ########################################
    # ###########################################################################
    def refinement_process_moea_qr(self, J, M, idx_ones, idx_zeroes):
        steps = 0

        # calculate theta and QR decomposition  O(p^2n)
        theta, q, r, r1 = self.theta_qr(J)

        # calculate rss  O(np)
        rss = self.rss(J, theta)

        # shortcut
        if M.shape[0] == 0:
            return self.Result(theta, idx_ones, rss, steps)

        while True:

            # calculate residuals r_1 ... r_n  O(np)
            residuals_J, residuals_M = self.all_residuals(J, M, theta)

            # find the optimal swap - j add ; i remove O(n^2p^2)
            i_swap, j_swap, rho = self.all_pairs_moea_qr(J, M, r1, rss, residuals_J, residuals_M)

            # strong necessary condition satisfied
            if rho >= 1:
                break

            else:

                # update rss
                rss = rss * rho

                # save row to swap in QR
                row_to_add = np.copy(M[j_swap, 1:])

                # swap observations
                self.swap_xi_xj(J, M, idx_ones, idx_zeroes, i_swap, j_swap)

                # update QR O(np^2) + O(n^2)
                q, r = self.qr_insert(q, r, row_to_add, i_swap + 1)
                q, r = self.qr_delete(q, r, i_swap)

                # update theta, r1  O(p^2)
                theta, r1 = self.theta_from_qr(q, r, J)
                steps += 1

        return self.Result(theta, idx_ones, rss, steps)

    # go through all combinations and find the optimal swap
    def all_pairs_moea_qr(self, J, M, R, rss, res_J, res_M):
        ro_min = 1
        i_swap = None
        j_swap = None

        # (moea speedup)
        ro_b_min = 1

        # calculate imi and jmj in advance O(p^2n)
        arr_imi, arr_vi = self.all_idx_m_idx_qr(M, R)  # for included rows
        arr_jmj, _ = self.all_idx_m_idx_qr(J, R)  # for excluded rows

        # go through all combinations
        for i in range(J.shape[0]):
            jmj = arr_jmj[i]

            for j in range(M.shape[0]):
                vi = arr_vi[j]
                imi = arr_imi[j]
                ei = res_M[j, 0]  # residual for included row
                ej = res_J[i, 0]  # residual for excluded row

                # calculate ro_b (moea speedup)
                a = ((1 + imi + (ei ** 2) / rss) * (1 - jmj - (ej ** 2) / rss))
                b = (1 + imi - jmj)
                ro_b = a / b
                if ro_b > ro_b_min:
                    continue

                # calculate ro_i_j multiplicative difference (Agullo)
                xj = J[i, 1:]
                u = linalg.solve_triangular(R, vi)
                i_m_j = np.dot(xj, u)
                i_m_j = i_m_j[0, 0]

                a = a + ((i_m_j + (ei * ej) / rss) ** 2)
                b = b + (i_m_j ** 2) - imi * jmj
                ro = a / b

                # update ro_b_min (moea speedup)
                if ro < ro_b_min:
                    ro_b_min = ro

                # update ro_min along with i j indexes
                if ro < ro_min:
                    ro_min = ro
                    i_swap = i
                    j_swap = j

        return i_swap, j_swap, ro_min

    @staticmethod
    def all_idx_m_idx_qr(M, R):
        arr_idx_m_idx = []
        arr_v_idx = []
        for j in range(M.shape[0]):
            x_idx = M[j, 1:]
            vi = linalg.solve_triangular(R.T, x_idx.T, lower=True)
            idx_m_idx = np.dot(vi.T, vi)  # vi.T * vi
            idx_m_idx = idx_m_idx[0, 0]

            arr_idx_m_idx.append(idx_m_idx)
            arr_v_idx.append(vi)

        return arr_idx_m_idx, arr_v_idx

    # ###########################################################################
    # ############### QR HELPERS ################################################
    # ###########################################################################

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

    # ###########################################################################
    # ###########################################################################
    # ###########################################################################

    # calculate the changes for all rows O(p^2n)
    @staticmethod
    def all_idx_m_idx_inv(M, inversion):
        arr_i_m_i = []
        for j in range(M.shape[0]):
            i_m_i = FSRegressor.dot_idx_m_idx(M, j, inversion)
            arr_i_m_i.append(i_m_i)
        return arr_i_m_i

    @staticmethod
    def dot_idx_m_idx(A, idx, inversion):
        x_idx = A[idx, 1:]
        idx_m_idx = np.dot(np.dot(x_idx, inversion), x_idx.T)
        idx_m_idx = idx_m_idx[0, 0]
        return idx_m_idx

    # ############### ALL PAIRS MOEA  ##################################################
    def all_pairs_moea_inv(self, J, M, inversion, rss, res_J, res_M):
        ro_min = 1
        i_swap = None
        j_swap = None

        # (moea speedup)
        ro_b_min = 1

        # calculate imi and jmj in advance O(p^2n)
        arr_imi = self.all_idx_m_idx_inv(M, inversion)  # for included rows
        arr_jmj = self.all_idx_m_idx_inv(J, inversion)  # for excluded rows

        # go through all combinations
        for i in range(J.shape[0]):
            jmj = arr_jmj[i]
            for j in range(M.shape[0]):
                imi = arr_imi[j]
                ei = res_M[j, 0]  # residual for included row
                ej = res_J[i, 0]  # residual for excluded row

                # calculate ro_b (moea speedup)
                a = ((1 + imi + (ei ** 2) / rss) * (1 - jmj - (ej ** 2) / rss))
                b = (1 + imi - jmj)
                ro_b = a / b
                if ro_b > ro_b_min:
                    continue

                # calculate ro_i_j multiplicative difference (Agullo)
                i_m_j = np.dot(np.dot(M[j, 1:], inversion), (J[i, 1:]).T)
                i_m_j = i_m_j[0, 0]
                a = a + ((i_m_j + (ei * ej) / rss) ** 2)
                b = b + (i_m_j ** 2) - imi * jmj
                ro = a / b

                # update ro_b_min (moea speedup)
                if ro < ro_b_min:
                    ro_b_min = ro

                # update ro_min along with i j indexes
                if ro < ro_min:
                    ro_min = ro
                    i_swap = i
                    j_swap = j

        return i_swap, j_swap, ro_min

    # ro_i_j multiplicative difference (Agullo)
    # @staticmethod
    # def ro_equation(rss, ei, ej, i_m_i, j_m_j, i_m_j):
    #     nn = (i_m_j + (ei*ej)/rss)
    #     nom = (1 + i_m_i + (ei**2)/rss)*(1 - j_m_j - (ej**2)/rss) + nn**2
    #     denom = (1 + i_m_i - j_m_j + (i_m_j**2) - i_m_i*j_m_j)
    #     ro = nom / denom
    #
    #     return ro

    # ###########################################################################
    # ############### DELTA RSS #################################################
    # ###########################################################################
    @staticmethod
    def calculate_delta_rss(J, M, inversion,
                            residuals_J, residuals_M, i, j):
        ei = residuals_J[i, 0]
        ej = residuals_M[j, 0]

        hii = J[i, 1:] * inversion * (J[i, 1:]).T  # 1xp * pxp * pX1
        hij = J[i, 1:] * inversion * (M[j, 1:]).T
        hjj = M[j, 1:] * inversion * (M[j, 1:]).T
        hii = hii[0, 0]
        hij = hij[0, 0]
        hjj = hjj[0, 0]

        nom = (ej * ej * (1 - hii)) - (ei * ei * (1 + hjj)) + 2 * ei * ej * hij
        denom = (1 - hii) * (1 + hjj) + hij * hij
        return nom / denom

    # ###########################################################################
    # ############### INITIAL H1 ################################################
    # ###########################################################################
    def generate_random_start(self, n):
        # create random permutation
        idx_all = np.random.permutation(n)
        # cut first h indexes and save the rest
        idx_initial = idx_all[:self._h_size]
        idx_rest = idx_all[self._h_size:]

        return idx_initial, idx_rest

    # ###########################################################################
    # ############### REFINEMENT MMEA INV #######################################
    # ###########################################################################
    def refinement_process_mmea_inv(self, J, M, idx_ones, idx_zeroes):
        steps = 0

        # calculate theta and inversion  O(p^2n)
        theta, inversion = self.theta_inv(J)

        # calculate rss
        rss = self.rss(J, theta)

        # shortcut
        if M.shape[0] == 0:
            return self.Result(theta, idx_ones, rss, steps)

        while True:

            # find optimal include  O(p^2n)
            j_swap, gamma_plus = self.smallest_include_inv(M, theta, inversion)

            # update theta -> theta_plus ; inversion -> inversion_plus  O(p^2)
            theta_plus, inversion_plus = self.theta_plus_inversion_plus(theta, inversion, M, j_swap)

            # find the optimal exclude (no need to update J ... worst case: gamma_plus == gamma_minus )  O(p^2n)
            i_swap, gamma_minus = self.greatest_exclude_inv(J, theta_plus, inversion_plus)

            # improvement cannot be made
            if not (gamma_plus < gamma_minus):
                break

            # update theta, inversion, rss, J, M, residualsJ residualsM

            # update rss
            rss = rss + gamma_plus - gamma_minus

            # update theta_plus -> theta_minus ; inversion_plus -> inversion_minus  O(p^2)
            theta, inversion = self.theta_minus_inversion_minus(theta_plus, inversion_plus, J, i_swap)

            # swap observations
            self.swap_xi_xj(J, M, idx_ones, idx_zeroes, i_swap, j_swap)
            steps += 1

        return self.Result(theta, idx_ones, rss, steps)

    # calculate gama plus  O(p^2)
    def gamma_plus_inv(self, M, j, inversion, theta):
        i_m_i = self.dot_idx_m_idx(M, j, inversion)
        xi = M[j, 1:]
        yi = M[j, [0]]  # 1 x 1
        yi_xi_theta = (yi - np.dot(xi, theta))[0, 0]
        gamma_plus = (yi_xi_theta ** 2) / (1 + i_m_i)
        return gamma_plus

    # calculate gamma minus  O(p^2)
    def gamma_minus_inv(self, J, i, inversion, theta):
        j_m_j = self.dot_idx_m_idx(J, i, inversion)
        xj = J[i, 1:]
        yj = J[i, [0]]  # 1 x 1
        yj_xj_theta = (yj - np.dot(xj, theta))[0, 0]
        gamma_minus = (yj_xj_theta ** 2) / (1 - j_m_j)
        return gamma_minus

    # find row which increase RSS the smallest if included O(p^2n)
    def smallest_include_inv(self, M, theta, inversion):
        gamma_plus_min = float('inf')
        idx_i = 0

        for j in range(M.shape[0]):

            # calculate rss change after including row j
            gamma_plus = self.gamma_plus_inv(M, j, inversion, theta)

            # and store the smallest
            if gamma_plus < gamma_plus_min:
                gamma_plus_min = gamma_plus
                idx_i = j

        return idx_i, gamma_plus_min

    # find row which reduce RSS the most if excluded O(p^2n)
    def greatest_exclude_inv(self, J, theta, inversion):
        gamma_minus_max = float('-inf')
        idx_j = 0

        for i in range(J.shape[0]):

            # calculate rss change after excluding row i
            gamma_minus = self.gamma_minus_inv(J, i, inversion, theta)

            # and store the greatest
            if gamma_minus > gamma_minus_max:
                gamma_minus_max = gamma_minus
                idx_j = i

        return idx_j, gamma_minus_max

    # #####################################################################################
    # ################# QR OPERATIONS #####################################################
    # #####################################################################################
    def qr_delete(self, q, r, idx):  # for j in (0, 1 ... n) * for i in (1, 2, ... n) ----> i guess O(n^2) == HODNE
        qnew = np.copy(q)
        rnew = np.copy(r)
        p_del = 1

        if idx != 0:  # swap it to the first line
            for j in range(idx, 0, -1):  # jdx ... 3, 2, 1
                qnew[[j, j - 1]] = qnew[[j - 1, j]]
                # swap(qnew[j,:], qnew[j-1, : ]) j jde od posledniho (od vlozeneho) a posouva ho na spravny index

        # for i in range p_del
        i = 0
        n = q.shape[0]
        p = r.shape[1]

        # od n-2 do 0 ( O(n) )
        for j in range(n - 2, i - 1, -1):  # n-2 protoze vzdy o jeden vic (tj. do idx n-1)

            cos, sin, R = self.calculate_cos_sin(qnew[0, j], qnew[0, j + 1])  # i, i = 0
            qnew[0, j] = R
            qnew[0, j + 1] = 0  # myslim ze pro nas nyni zbytecne ?

            # update rows to del - no need
            # if i + 1 < p_del: #  1 < 1
            #     rot(p - i - 1, index2(W, ws, i + 1, j), ws[0],
            #         index2(W, ws, i + 1, j + 1), ws[0], c, s)

            # O(p) asi ...
            # Rotare R if nonzero row
            if j < p:  # m x n # j - i

                # tod o rot( [ p-j-1 ]  [--, j+1 ] [--, j+1]
                # to znamena naky radky...
                # a v kazdem du od j+1 do konce

                # rotate rnew
                rowX = rnew[j, :]
                rowY = rnew[j + 1, :]
                # blas srot
                for i in range(j, p):  # vzdy od j do p konce # mozna j+1; ne myslim ze ok
                    temp = cos * rowX[i] + sin * rowY[i]
                    rnew[j + 1, i] = cos * rowY[i] - sin * rowX[i]  # Y
                    rnew[j, i] = temp  # X

            # 1 ... n O(n)
            # Rotate Q - pozor - fucking TRICK qs[0]
            qcolX = qnew[:, j]
            qcolY = qnew[:, j + 1]
            for i in range(p_del, n):  # radky 1, 2, ... n-1 #
                temp = cos * qcolX[i] + sin * qcolY[i]  # here was error
                qnew[i, j + 1] = cos * qcolY[i] - sin * qcolX[i]  # Y
                qnew[i, j] = temp  # X

        return qnew[p_del:, p_del:], rnew[p_del:, :]

    def qr_insert(self, q, r, row, idx):  # O(p * n)
        # idx .. row before which new row will be inserted
        n = q.shape[0]  # rows
        p = r.shape[1]  # cols
        cnt_rows = 1

        shape = [n + cnt_rows, n + cnt_rows]

        # create new matrix
        qnew = np.zeros(shape, dtype=float)
        shape[1] = p
        rnew = np.zeros(shape, dtype=float)

        # fill matrix r
        rnew[:n, :] = np.copy(r)
        rnew[n:, :] = row  # just one row in this case (1,7 into 1,6)

        # fill matrix q
        qnew[:-cnt_rows, :-cnt_rows] = q
        for j in range(n, n + cnt_rows):  # again, neni treba loop; jen posledni radek
            qnew[j, j] = 1

        n = n + 1
        # rotate last row and update both matrix
        limit = min(n - 1, p)  # opet, autimaticky je to p ;  n-1 kvuli poslednimu sloupku q ?

        # we are basically removing just from Q
        for j in range(limit):  # pro kazdy element posledniho radku (p)
            cos, sin, R = self.calculate_cos_sin(rnew[j, j], rnew[n - 1, j])  # edge of triangle , last row
            rnew[j, j] = R  # some hack as they have
            rnew[n - 1, j] = 0

            # rotate rnew
            rowX = rnew[j, :]  # row
            rowY = rnew[n - 1, :]  # row
            # blas srot
            for i in range(j + 1, p):  # vzdy od j do konce (udelej rotaci celeho radku)
                temp = cos * rowX[i] + sin * rowY[i]
                rnew[n - 1, i] = cos * rowY[i] - sin * rowX[i]  # Y
                rnew[j, i] = temp  # X

            # rotate qnew
            q_rowX = qnew[:, j]  # j ty slouepk (delky n)
            q_rowY = qnew[:, n - 1]  # posledni slopek (delky n)

            # blas srot
            for i in range(n):  # vzdy od j do konce
                temp = cos * q_rowX[i] + sin * q_rowY[i]
                qnew[i, n - 1] = cos * q_rowY[i] - sin * q_rowX[i]  # Y
                qnew[i, j] = temp  # X

        # move last (inserted) row to correct position
        # k je jako idx ? jo. je to jako idx v nove matici
        # chci ho hodit za ten co potom odstranim (teda idx bude prvdepodobne idxj+1)
        for j in range(n - 1, idx, -1):
            # this is bad - it is copy swap !!!
            qnew[[j - 1, j]] = qnew[[j, j - 1]]
            # swap(q new[j,:], q new[j-1, : ]) j jde od posledniho (od vlozeneho) a posouva ho na spravny index

        return qnew, rnew

    # lapack  slartg
    @staticmethod
    def calculate_cos_sin(f, g):

        if g == 0:
            cos = 1
            sin = 0
            r = f
        elif f == 0:
            cos = 0
            sin = 1
            r = g
        else:
            r = math.sqrt(f ** 2 + g ** 2)
            cos = f / r
            sin = g / r

        if abs(f) > abs(g) and cos < 0:
            cos = -cos
            sin = -sin
            r = -r

        return cos, sin, r
