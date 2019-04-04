from lts.feasible.helpers import AbstractRegression
from lts.feasible.helpers import validate
import numpy as np
import math
import time
# import cppimport.import_hook
# import lts.feasibleoe.cpp.feasible_solution as cpp_solution

from sklearn.linear_model import LinearRegression

from scipy import linalg

"""
# lts = cppimport.imp("feasible/cpp/feasible_solution")
cppimport
    For cppimport adhere this naming convention:
    cpp file: xxx.cpp
        inside: PYBIND11_MODULE(xxx, m)
    inside python module: my_import =  cppimport.imp("xxx")
"""

# class FSRegressorCPP(AbstractRegression):
#     def __init__(self):
#         super().__init__()
#         self._data = None
#         self._p = None
#         self._N = None
#         self._h_size = None
#         # public
#         self.n_iter_ = None
#         self.coef_ = None
#         self.intercept_ = None
#         self.h_subset_ = None
#         self.rss_ = None
#         self.time1_ = None
#         self.time_total_ = None
#
#     # ############### FIT #######################################################
#     def fit(self, X, y,
#             num_starts: 'number of initial starts (H1)' = 10,
#             h_size: 'default := (n + p + 1) / 2' = 'default',
#             use_intercept=True):
#
#         X, y = validate(X, y, h_size, use_intercept)
#
#         # todo h_size including intercept?
#         h_size = math.ceil((X.shape[0] + X.shape[1] + 1) / 2) if h_size == 'default' else h_size  # N + p + 1
#
#         result = cpp_solution.fs_lts(X, y, num_starts, h_size)
#
#         # Store result - weights first
#         weights = result.get_theta()
#         if use_intercept:
#             self.intercept_ = weights[-1, 0]  # last row first col
#             self.coef_ = np.ravel(weights[:-1, 0])  # for all but last column,  only first col
#         else:
#             self.intercept_ = 0.0
#             self.coef_ = np.ravel(weights[:, 0])  # all rows, only first col
#
#         # Store rest of the attributes
#         self.h_subset_ = result.get_h_subset()
#         self.rss_ = result.get_rss()
#         self.n_iter_ = result.get_n_inter()
#         self.time1_ = result.get_time_1()
#         self.time_total_ = self.time1_


class FSRegressor(AbstractRegression):
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

    def fit_bab(self, X, y, h_size: 'default := (n + p + 1) / 2' = 'default', use_intercept=True):
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
            
        J = np.matrix(self._data, copy=True)

        best_result = self.bab_lts(J, self._h_size)

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


    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 10,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True):

        # TODO - bab lts
        time1 = time.process_time()
        self.fit_bab(X, y)
        # save the time
        self.time1_ = time.process_time() - time1
        self.time_total_ = self.time1_
        return

        # todo bab lts

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
            # save split data
            J = np.matrix(self._data[idx_initial], copy=True)
            M = np.matrix(self._data[idx_rest], copy=True)

            # J2 = np.copy(J)
            # M2 = np.copy(M)
            # idx_initial2 = np.copy(idx_initial)
            # idx_rest2 = np.copy(idx_rest)
            # J2 = np.asmatrix(J2)
            # M2 = np.asmatrix(M2)
            # do the refinement process
            res = self.refinement_process_fs_mmea_inversion(J, M, idx_initial, idx_rest)
            # print(res.rss)
            #
            # res = self.refinement_process_fsa(J, M, idx_initial, idx_rest)
            # print(res.steps)
            # print(res.rss)
            # exit(1)
            # create new J and M so that it represents h_index
            # res = self.refinement_process_fsa_MMEA_qr(J, M, idx_initial, idx_rest)
            # res2 = self.refinement_process_fsa(J2, M2, idx_initial2, idx_rest2)

            # todo - porovnani results

            # print('**** RESULT *******')
            # print('theta ')
            # print(res.theta_hat)
            # print('theta 2')
            # print(res2.theta_hat)
            # if not (math.isclose(res.rss, res2.rss)):
            #     print('rss [{}], [{}]'.format(res.rss, res2.rss))
            #     print('steps [{}], [{}]'.format(res.steps, res2.steps))
            #     exit(1)
            # print('OK')

            results.append(res)

        # save the time
        self.time1_ = time.process_time() - time1
        self.time_total_ = self.time1_

        # select best results
        best_result = results[0]
        for res in results:
            if res.rss < best_result.rss:
                best_result = res


        # # todo - just some experiment
        #
        # print('prvni alg hotvo. dalsich steps {} ; rss {}'.format(best_result.steps, best_result.rss))
        # print('jdu na druhy alg.')
        # idx_initial = best_result.h_index
        #
        # mask = np.ones([self._data.shape[0]], np.bool)
        # mask[idx_initial] = 0
        # idx_rest = mask
        #
        # J = np.matrix(self._data[idx_initial], copy=True)
        # M = np.matrix(self._data[idx_rest], copy=True)
        #
        # best_result = self.refinement_process_fs_moe_inversion(J, M, idx_initial, idx_rest)
        # print('hotovo. dalsich steps {} ; rss {}'.format(best_result.steps, best_result.rss))
        # # todo - just some experiment

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
    # ############### FSA VERSION  ##############################################
    # ###########################################################################
    def refinement_process_fsa(self, J, M, idx_initial, idx_rest):
        steps = 0
        while True:

            # Calculate theta and store inversion
            theta, inversion = self.calculate_theta_inversion(J)

            # Calculate residuals
            residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

            # Iterate all swap combinations
            i_to_swap, j_to_swap, delta = self.all_pairs_fsa(J, M, inversion, residuals_J, residuals_M)

            if delta >= 0:
                break

            else:  # swap i and j (together with indexes)
                self.swap_row_J_M(J, M, idx_initial, idx_rest, i_to_swap, j_to_swap)
                steps += 1

        # Save converged result
        y_fin = J[:, [0]]
        x_fin = J[:, 1:]
        rss = (y_fin - x_fin * theta).T * (y_fin - x_fin * theta)
        rss = rss[0, 0]

        return self.Result(theta, idx_initial, rss, steps)

    def all_pairs_fsa(self, J, M, inversion, residuals_J, residuals_M):  # todo - odmazat delta 2
        delta = 0
        i_to_swap = None
        j_to_swap = None

        # go through all combinations
        for i in range(J.shape[0]):
            for j in range(M.shape[0]):
                # . calculate deltaRSS
                tmp_delta = self.calculate_delta_rss(J, M, inversion, residuals_J, residuals_M, i, j)

                # if delta rss < bestDeltaRss
                if tmp_delta < delta:
                    # print('**************')
                    # print(tmp_delta)
                    # print(delta)
                    # print('---')
                    delta = tmp_delta
                    # print(delta)
                    # print('**************')
                    i_to_swap = i
                    j_to_swap = j

        return i_to_swap, j_to_swap, delta

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
    # ############### FSA-OE-QR VERSION  ########################################
    # ###########################################################################

    # Calculate theta using normal equation: R1 theta = Q1y
    def calculate_theta_qr(self, J):
        # # Q ... n x n
        # # R ... n x p
        q, r = linalg.qr(J[:, 1:])  # X = QR ; x.T x = R.T R ;
        # ( connection to Cholesky: L * L.T  where l = R.T)

        # #  Q.T *  ( x * w - y ) ^ 2
        # #  Q.T * Q * R * w - Q.T * y
        # #  R * w - Q.T * y
        # #  R * w = Q.T * y
        theta, r1 = self.update_theta_qr(q, r, J)

        return theta, q, r, r1

    def calculate_theta_inversion(self, J):
        y = J[:, [0]]
        x = J[:, 1:]

        inversion = (x.T * x).I
        theta = inversion * x.T * y  # OLS
        return theta, inversion

    # Update theta using normal equation: R1 theta = Q1y
    def update_theta_qr(self, q, r, J):
        y = J[:, [0]]
        p = r.shape[1]
        #  r1 pxp
        r1 = r[:p, :]  # only first p rows
        qt = q.T
        q1 = qt[:p, :]  # only first p rows

        # # Solve the equation x w = c for x, assuming a is a triangular matrix
        theta = linalg.solve_triangular(r1, q1 * y)  # p x substitution
        return theta, r1

    # Calculate theta directly from ~M without c = Q1y
    def calculate_theta_fii(self, Ja):
        J = np.copy(Ja)
        J = np.asmatrix(J)

        # move first col (y) to last col so we'll have (X|y)
        first_col = J[:, [0]]
        J[:, : -1] = J[:, 1:]
        J[:, [-1]] = first_col

        qM, rM = linalg.qr(J)
        theta, rss, r1 = self.update_theta_fii(rM)

        return theta, qM, rM, r1, rss

    # Update theta directly from ~M without c = Q1y
    def update_theta_fii(self, rM):
        p = rM.shape[1] - 1
        r1 = rM[:p, : -1]
        fii = rM[:p, [-1]]
        theta = linalg.solve_triangular(r1, fii)
        rss = rM[p, p] ** 2

        return theta, rss, r1

    # calculate residuals from all used and unused observations
    def calculate_residuals_y_first(self, J, M, theta):
        residuals_J = J[:, [0]] - J[:, 1:] * theta
        residuals_M = (M[:, [0]]) - (M[:, 1:]) * theta

        return residuals_J, residuals_M

    def swap_row_J_M(self, J, M, idx_initial, idx_rest, i_to_swap, j_to_swap):
        tmp = np.copy(J[i_to_swap])
        J[i_to_swap] = np.copy(M[j_to_swap])
        M[j_to_swap] = np.copy(tmp)
        idx_initial[i_to_swap], idx_rest[j_to_swap] = idx_rest[j_to_swap], idx_initial[i_to_swap]

    def test_qr_decomp(self, q, r, X):

        #
        # if not np.allclose(np.dot(qM.T, qM), np.eye(what_we_want.shape[0])):
        #     print('not orthogonal Q')
        #     exit(1)

        # # if not np.allclose(np.dot(q3.T, q3), np.eye(what_we_want.shape[0])):
        # #     print('not orthogonal Q3')
        # #     exit(1)
        #
        # if not np.allclose(np.dot(qM, rM), what_we_want):
        #     print('not similar Q R ')
        #     exit(1)

        # if not np.allclose(np.dot(q3, r3), what_we_want):
        #     print('not similar Q3 R3 ')
        #     exit(1)

        # r1 = r[:p, :]  # only first p rows
        # theta, r1 = self.update_theta_qr(q, r, J)

        # if not np.allclose(theta, theta2):
        #     print(theta)
        #     print(theta2)
        #     print('********')
        #     exit(1)
        #
        # if not math.isclose(rss[0, 0], rss2, rel_tol=1e-09):
        #     print(rss)
        #     print(rss2)
        #     exit(1)

        return

    def refinement_process_fs_moe_qr(self, J, M, idx_initial, idx_rest):
        steps = 0

        # Calculate QR decompositon
        theta, q, r, r1 = self.calculate_theta_qr(J)

        # Calculate residuals e
        residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

        # Calculate RSS
        rss = residuals_J.T * residuals_J
        rss = rss[0, 0]
        while True:
            i_to_swap, j_to_swap, delta = self.all_pairs_fsa_oe_qr(J, M, r1, rss, residuals_J, residuals_M)

            if delta >= 1:
                break

            else:
                # Save row to swap in QR
                row_to_add = np.copy(M[j_to_swap, 1:])

                # Update J and M arrays and also idx array by means of swapped rows
                self.swap_row_J_M(J, M, idx_initial, idx_rest, i_to_swap, j_to_swap)

                # Update RSS
                rss = rss * delta

                # Update QR
                q, r = self.qr_insert(q, r, row_to_add, i_to_swap + 1)
                q, r = self.qr_delete(q, r, i_to_swap)

                # Update Theta, R1
                theta, r1 = self.update_theta_qr(q, r, J)

                # calculate residuals M and J
                residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

                steps += 1

        return self.Result(theta, idx_initial, rss, steps)

    def greatest_remove_mmea_inversion(self, J, theta, inversion):
        gama_remove_max = float('-inf')
        j_m_j_min = 0
        yj_xj_theta_min = 0
        idx_j = 0

        for i in range(J.shape[0]):
            xj = J[i, 1:]
            yj = J[i, [0]]  # 1 x 1
            j_m_j = np.dot(np.dot(xj, inversion), xj.T)
            j_m_j = j_m_j[0, 0]
            yj_xj_theta = (yj - np.dot(xj, theta))[0, 0]
            gama_remove = (yj_xj_theta **2) / (1 - j_m_j)

            if gama_remove > gama_remove_max:
                gama_remove_max = gama_remove
                j_m_j_min = j_m_j
                yj_xj_theta_min = yj_xj_theta
                idx_j = i

        return idx_j, j_m_j_min, yj_xj_theta_min, gama_remove_max

    def smallest_insert_mmea_inversion(self, M, theta, inversion):
        gama_insert_min = float('inf')
        i_m_i_min = 0
        yi_xi_theta_min = 0
        idx_i = 0

        for j in range(M.shape[0]):
            # Calculate gama plus
            xi = M[j, 1:]
            yi = M[j, [0]]  # 1 x 1
            i_m_i = np.dot(np.dot(xi, inversion), xi.T)
            i_m_i = i_m_i[0, 0]
            yi_xi_theta = (yi - np.dot(xi, theta))[0, 0]
            gama_insert = (yi_xi_theta ** 2 ) / (1 + i_m_i)

            # save is smallest
            if gama_insert < gama_insert_min:
                gama_insert_min = gama_insert
                i_m_i_min = i_m_i
                yi_xi_theta_min = yi_xi_theta
                idx_i = j

        return idx_i, i_m_i_min, yi_xi_theta_min, gama_insert_min

    def refinement_process_fs_mmea_inversion(self, J, M, idx_initial, idx_rest):
        steps = 0

        # Calculate theta and store inversion
        theta, inversion = self.calculate_theta_inversion(J)

        # Calculate residuals
        residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

        # Calculate RSS
        rss = (residuals_J.T * residuals_J)[0, 0]

        # OLS shortcut
        if M.shape[0] == 0:
            return self.Result(theta, idx_initial, rss, steps)

        while True:

            # find the insert
            # todo check correct
            idx_insert, i_m_i, yi_xi_theta, gamma_plus = self.smallest_insert_mmea_inversion(M, theta, inversion)

            #
            # Theta plus
            xi = M[idx_insert, 1:]  # 1 x p
            w = -1 / (1 + i_m_i)  # 1x1
            u = np.dot(inversion, xi.T)  # p x 1
            theta_plus = theta + (-1 * yi_xi_theta * w * u)  # todo OK !!!! (changed [* -1] )

            #
            # Inversion plus
            inversion_plus = inversion + w * np.dot(u, u.T)  # todo OK

            #
            # find the remove
            # no need to update J - no need for removing the same.. then gama_plus == gama_minus
            idx_remove, j_m_j, yj_xj_theta_min, gamma_minus = self.greatest_remove_mmea_inversion(J, theta_plus, inversion_plus)

            #
            # edit J, M, inversion, rss, residualsJ residualsM
            if not (gamma_plus < gamma_minus):
                break

            #
            # Update RSS
            rss = rss + gamma_plus - gamma_minus

            #
            # Theta plus minus
            xj = J[idx_remove, 1:]
            wj = -1 / (1 - j_m_j)
            uj = np.dot(inversion_plus, xj.T)
            theta_plus_minus = theta_plus + (yj_xj_theta_min * (wj * uj))  # todo  nepresen e-5

            #
            # Inversion plus minus
            inversion_plus_minus = inversion_plus - wj * np.dot(uj,
                                                                uj.T)  # sloupek * radek = matice todo - nepresne e-9
            theta = theta_plus_minus
            inversion = inversion_plus_minus

            #
            # Update J and M arrays and also idx array by means of swapped rows
            self.swap_row_J_M(J, M, idx_initial, idx_rest, idx_remove, idx_insert)

            steps += 1

        return self.Result(theta, idx_initial, rss, steps)

    def refinement_process_fs_moe_inversion(self, J, M, idx_initial, idx_rest):
        steps = 0

        # Calculate theta and store inversion
        theta, inversion = self.calculate_theta_inversion(J)

        # Calculate residuals
        residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

        # Calculate RSS
        rss = (residuals_J.T * residuals_J)[0, 0]

        while True:
            i_to_swap, j_to_swap, delta, i_m_i, j_m_j = self.all_pairs_fsa_oe_inversion(J, M, inversion, rss, residuals_J, residuals_M)

            if delta >= 1:
                break
            else:

                # Update RSS
                rss = rss * delta

                # Update Theta and Inversion ********************************************************************

                #
                # Theta plus
                xi = M[j_to_swap, 1:]  # 1 x p
                yi = M[j_to_swap, [0]]  # 1 x 1
                w = -1 / (1 + i_m_i)  # 1x1
                u = np.dot(inversion, xi.T)  # p x 1
                theta_plus = theta + (-1* (yi - np.dot(xi, theta))[0, 0] * (w * u))#  todo OK !!!! (changed [* -1] )

                # xx = J[:, 1:]  # 1 x p
                # yy = J[:, [0]]  # 1 x 1
                # theta_plus = ( inversion + w * np.dot(u, u.T) )  * ( xx.T * yy + np.dot( yi[0,0], xi.T ) )
                # Jplus = np.append(J,  M[j_to_swap,:], axis=0)
                # theta_test, inversion_test = self.calculate_theta_inversion(Jplus)
                #
                # if not np.allclose(theta_test, theta_plus):
                #     print('not similar theta plus ')
                #     print(np.abs(theta) - np.abs(theta_test))
                #     print('----')
                #     print(np.abs(theta) - np.abs(theta_plus))
                #     exit(1)
                # exit(1)

                #
                # Inversion plus
                inversion_plus = inversion + w * np.dot(u, u.T)  # todo OK

                #
                # Theta plus minus
                xj = J[i_to_swap, 1:]
                yj = J[i_to_swap, [0]]
                wj = -1 / (1 - j_m_j)
                uj = np.dot(inversion_plus, xj.T)
                theta_plus_minus = theta_plus + ( (yj - np.dot(xj, theta_plus))[0, 0] * (wj * uj) )  #todo  nepresen e-5

                #
                # Inversion plus minus
                inversion_plus_minus = inversion_plus - wj * np.dot(uj, uj.T)  # sloupek * radek = matice todo - nepresne e-9
                theta = theta_plus_minus
                inversion = inversion_plus_minus

                #
                # Update J and M arrays and also idx array by means of swapped rows
                self.swap_row_J_M(J, M, idx_initial, idx_rest, i_to_swap, j_to_swap)

                residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

                steps += 1
        return self.Result(theta, idx_initial, rss, steps)

    def refinement_process_fs_moe_qr_extended(self, J, M, idx_initial, idx_rest):
        steps = 0

        # Calculate QR decompositon along with theta and RSS directly from (X|y)
        theta, qM, rM, r1, rss = self.calculate_theta_fii(J)

        # Calculate residuals e
        residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

        while True:
            i_to_swap, j_to_swap, delta = self.all_pairs_fsa_oe_qr(J, M, r1, rss, residuals_J, residuals_M)

            if delta >= 1:
                break

            else:
                # Save row to swap in QR
                row_to_addM = np.copy(M[j_to_swap, :])
                # move first elem (y) to last col
                first_col = row_to_addM[:, [0]]
                row_to_addM[:, : -1] = row_to_addM[:, 1:]
                row_to_addM[:, [-1]] = first_col

                # Update J and M arrays and also idx array by means of swapped rows
                self.swap_row_J_M(J, M, idx_initial, idx_rest, i_to_swap, j_to_swap)

                # Update QR
                qM, rM = self.qr_insert(qM, rM, row_to_addM, i_to_swap + 1)
                qM, rM = self.qr_delete(qM, rM, i_to_swap)

                # Update theta, R1, RSS
                theta, rss, r1 = self.update_theta_fii(rM)

                # calculate residuals M and J
                residuals_J, residuals_M = self.calculate_residuals_y_first(J, M, theta)

                steps += 1

        return self.Result(theta, idx_initial, rss, steps)

    def all_pairs_fsa_oe_inversion(self, J, M, inversion, rss, residuals_J, residuals_M):
        delta = 1
        i_to_swap = None
        j_to_swap = None
        im = None
        jm = None

        # Because of MOEA speedup
        ro_min = 1

        # Calculate all insertions i_m_i in advance e.g. all added rows
        arr_i_m_i = []
        for j in range(M.shape[0]):
            xi = M[j, 1:]
            i_m_i = np.dot(np.dot(xi, inversion), xi.T)
            i_m_i = i_m_i[0, 0]
            arr_i_m_i.append(i_m_i)

        # go through all Combinations
        for i in range(J.shape[0]):
            # Calculate j_m_j e.g. removed row
            xj = J[i, 1:]
            j_m_j = np.dot(np.dot(xj, inversion), xj.T)
            j_m_j = j_m_j[0, 0]

            for j in range(M.shape[0]):  # this runs often, have prepared residuals_M
                # Retrieve e_i and e_j (Pre-calculated)
                ei = residuals_M[j, 0]
                ej = residuals_J[i, 0]

                # Calculate ro_b (because of MOEA speedup)
                ro_b = ((1 + arr_i_m_i[j] + rss / (ei ** 2)) * (1 - j_m_j - rss / (ej ** 2))) / (
                        1 + arr_i_m_i[j] - j_m_j)
                if ro_b > ro_min:
                    continue

                # Calculate true ro_rss
                i_m_j = np.dot(np.dot(M[j, 1:], inversion), xj.T)
                i_m_j = i_m_j[0, 0]
                tmp_delta = self.ro_equation(rss, ei, ej, arr_i_m_i[j], j_m_j, i_m_j)

                # Update ro_min if necessary (MOEA)
                if tmp_delta < ro_min:
                    ro_min = tmp_delta

                # Save smallest delta along with i j positions
                if tmp_delta < delta:
                    delta = tmp_delta
                    i_to_swap = i
                    j_to_swap = j
                    im = arr_i_m_i[j]
                    jm = j_m_j

        return i_to_swap, j_to_swap, delta, im, jm

    def ro_equation(self, rss, ei, ej, i_m_i, j_m_j, i_m_j):
        # Now just calculate the fraction
        nom = (1 + i_m_i + 1 / rss * ei ** 2) * (1 - j_m_j - 1 / rss * ej ** 2) + (i_m_j + 1 / rss * ei * ej) * (
                i_m_j + 1 / rss * ei * ej)
        denom = (1 + i_m_i - j_m_j + i_m_j ** 2 - i_m_i * j_m_j)
        ro = nom / denom

        return ro

    # Go through all combinations (all pairs of Ji and Mj) and calculate delta RSS
    def all_pairs_fsa_oe_qr(self, J, M, R, rss, residuals_J, residuals_M):
        delta = 1
        i_to_swap = None
        j_to_swap = None
        # Because of MOEA speedup
        ro_min = 1

        # Calculate all insertions i_m_i in advance e.g. all added rows
        # Along with all v_i
        arr_vi = []
        arr_i_m_i = []
        for j in range(M.shape[0]):
            xi = M[j, 1:]
            vi = linalg.solve_triangular(R.T, xi.T, lower=True)
            i_m_i = np.dot(vi.T, vi)  # vi.T * vi
            i_m_i = i_m_i[0, 0]
            arr_vi.append(vi)
            arr_i_m_i.append(i_m_i)

        # go through all Combinations
        for i in range(J.shape[0]):

            # Calculate j_m_j e.g. removed row
            xj = J[i, 1:]
            vj = linalg.solve_triangular(R.T, xj.T, lower=True)
            j_m_j = np.dot(vj.T, vj)
            j_m_j = j_m_j[0, 0]

            for j in range(M.shape[0]):  # this runs often, have prepared residuals_M

                # Retrieve e_i and e_j (Pre-calculated)
                ei = residuals_M[j, 0]
                ej = residuals_J[i, 0]

                # Calculate ro_b (because of MOEA speedup)
                ro_b = ((1 + arr_i_m_i[j] + rss / (ei ** 2)) * (1 - j_m_j - rss / (ej ** 2))) / (
                            1 + arr_i_m_i[j] - j_m_j)
                if ro_b > ro_min:
                    continue

                # Calculate true ro_rss
                tmp_delta = self.calculate_delta_rss_oe_qr(J, M, R, rss, ei, ej, i, j, arr_vi[j], arr_i_m_i[j], j_m_j,
                                                           xj)

                # Update ro_min if necessary (MOEA)
                if tmp_delta < ro_min:
                    ro_min = tmp_delta

                # Save smallest delta along with i j positions
                if tmp_delta < delta:
                    delta = tmp_delta
                    i_to_swap = i
                    j_to_swap = j

        return i_to_swap, j_to_swap, delta

    def calculate_delta_rss_oe_qr(self, J, M, R1, rss, ei, ej, i, j, vi, i_m_i, j_m_j, xj):
        # Calculate i_m_j e.g. in-out
        # xj * (R.T * R ) ^-1 * xi = xj * u.T
        #  R * u.T = vi ;
        #  where vi solves  R.T * vi.T = xi.T
        u = linalg.solve_triangular(R1, vi)
        i_m_j = np.dot(xj, u)
        i_m_j = i_m_j[0, 0]

        # Now just calculate the fraction
        ro = self.ro_equation(rss, ei, ej, i_m_i, j_m_j, i_m_j)
        return ro

    # ############### QR OPERATIONS #############################################
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

        for j in range(n - 2, i - 1, -1):  # n-2 protoze vzdy o jeden vic (tj. do idx n-1)

            cos, sin, R = self.calculate_cos_sin(qnew[0, j], qnew[0, j + 1])  # i, i = 0
            qnew[0, j] = R
            qnew[0, j + 1] = 0  # myslim ze pro nas nyni zbytecne ?

            # update rows to del - no need
            # if i + 1 < p_del: #  1 < 1
            #     rot(p - i - 1, index2(W, ws, i + 1, j), ws[0],
            #         index2(W, ws, i + 1, j + 1), ws[0], c, s)

            # Rotare R if nonzero row
            if j < p:  # m x n # j - i

                # todo rot( [ p-j-1 ]  [--, j+1 ] [--, j+1]
                # to znamena naky radky...
                # a v kazdem du od j+1 do konce

                # rotate rnew
                rowX = rnew[j, :]
                rowY = rnew[j + 1, :]
                # blas srot
                for i in range(j, p):  # vzdy od j do p konce /todo mozna j+1; ne myslim ze ok
                    temp = cos * rowX[i] + sin * rowY[i]
                    rnew[j + 1, i] = cos * rowY[i] - sin * rowX[i]  # Y
                    rnew[j, i] = temp  # X

            # Rotate Q - pozor - fucking TRICK qs[0]
            qcolX = qnew[:, j]
            qcolY = qnew[:, j + 1]
            for i in range(p_del, n):  # radky 1, 2, ... n-1 #
                temp = cos * qcolX[i] + sin * qcolY[i]  # todo chyba
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
            rowX = rnew[j, :]
            rowY = rnew[n - 1, :]
            # blas srot
            for i in range(j + 1, p):  # vzdy od j do konce (udelej rotaci celeho radku)
                temp = cos * rowX[i] + sin * rowY[i]
                rnew[n - 1, i] = cos * rowY[i] - sin * rowX[i]  # Y
                rnew[j, i] = temp  # X

            # rotate qnew
            qrowX = qnew[:, j]  # j ty slouepk
            qrowY = qnew[:, n - 1]  # posledni slopek

            # blas srot
            for i in range(n):  # vzdy od j do konce
                temp = cos * qrowX[i] + sin * qrowY[i]
                qnew[i, n - 1] = cos * qrowY[i] - sin * qrowX[i]  # Y
                qnew[i, j] = temp  # X

        # move last (inserted) row to correct position
        # k je jako idx ? jo. je to jako idx v nove matici
        # chci ho hodit za ten co potom odstranim (teda idx bude prvdepodobne idxj+1)
        for j in range(n - 1, idx, -1):
            # this is bad - it is copy swap !!!
            qnew[[j - 1, j]] = qnew[[j, j - 1]]
            # swap(qnew[j,:], qnew[j-1, : ]) j jde od posledniho (od vlozeneho) a posouva ho na spravny index

        return qnew, rnew

    # lapack  slartg
    def calculate_cos_sin(self, f, g):

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

    def bab_lts(self, J, h_size):
        if h_size == 0 or h_size == 1 :
            print('h_size must be at least 2')
            exit(1)

        self.bab_rss_min = float('inf')
        self.bab_indexes = None
        self.bab_theta = None
        a = []
        b = list(range(J.shape[0]))
        self.J = J
        self._h_size = h_size
        self.traverse_recursive(a, b, 0, None, None, None)

        steps = 0
        return self.Result(self.bab_theta, self.bab_indexes, self.bab_rss_min, steps)

    def calculate_gama_insert(self, idx, inversion, theta):
        xi = self.J[idx, 1:]
        yi = self.J[idx, [0]]  # 1 x 1
        i_m_i = np.dot(np.dot(xi, inversion), xi.T)
        i_m_i = i_m_i[0, 0]
        yi_xi_theta = (yi - np.dot(xi, theta))[0, 0]
        gama_insert = (yi_xi_theta ** 2) / (1 + i_m_i)

        return gama_insert, i_m_i, yi_xi_theta

    def calculate_theta_inversion_with_rss(self, J):
        y = J[:, [0]]
        x = J[:, 1:] # n x p

        inversion = (x.T * x).I # p x p
        theta = inversion * x.T * y  # OLS p x p * p  x n == p x p * (1xp) tak sme pekne v pici

        y = J[:, 0]
        x = J[:, 1:]

        nasob = x * theta
        residuals = J[:, 0] - nasob
        rss = (residuals.T * residuals)[0, 0]

        return theta, inversion, rss


    def calculate_theta_inversion_qr_with_rss(self, J):
        y = J[:, [0]]
        x = J[:, 1:] # n x p

        inversion = (x.T * x).I # p x p
        theta = inversion * x.T * y  # OLS p x p * p  x n == p x p * (1xp) tak sme pekne v pici

        y = J[:, 0]
        x = J[:, 1:]

        nasob = x * theta
        residuals = J[:, 0] - nasob
        rss = (residuals.T * residuals)[0, 0]

        return theta, inversion, rss

    def traverse_recursive(self, a, b, depth, rss, theta, inversion):

        # LEAF - RETURN
        if depth == self._h_size:
            #print('\t leaf {} ; l={}'.format(a, depth))

            # Calculate gama plus and new RSS
            gama_insert, i_m_i, yi_xi_theta = self.calculate_gama_insert(a[-1], inversion, theta)
            rss_here = rss + gama_insert

            # Possibly update result
            if rss_here < self.bab_rss_min:
                self.bab_rss_min = rss_here
                self.bab_indexes = np.copy(a)

                # todo - update theta and save it also - DONE
                xi = self.J[a[-1], 1:]
                w = -1 / (1 + i_m_i)
                u = np.dot(inversion, xi.T)
                theta_here = theta + (-1 * yi_xi_theta * w * u)  # todo OK !!!! (changed [* -1] )

                self.bab_theta = theta_here

            #print('---------------------------------------')
            #print('updated rss {}'.format(rss_here))
            #print('indexes here')
            #print(a)




            # todo - only testing
            # y = self.J[a, 0]
            # X = self.J[a, 1:]
            #
            # reg = LinearRegression(fit_intercept=False).fit(X, y)
            #
            # print('scikit rss {}'.format(reg.score(X, y)))
            # # reg.coef_
            # # reg.intercept_
            #
            # theta, _, _, _ = self.calculate_theta_qr(self.J[a])
            #
            # y = self.J[a, 0]
            # x = self.J[a, 1:]
            #
            # nasob = x * theta
            # residuals = y - nasob
            # rss = (residuals.T * residuals)[0, 0]
            #
            # #_, _, rss = self.calculate_theta_inversion_with_rss(self.J[a])
            # print('manual rss: {}'.format(rss))
            #
            # theta= (reg.coef_).T
            # nasob = x * theta
            # residuals = y - nasob
            # rss = (residuals.T * residuals)[0, 0]
            # print('manual scikit rss: {}'.format(rss))
            #
            # exit(1)

            # todo - only testig

            return

        # LEAF - FAKE - EXIT
        if len(b) == 0:
            exit(3)

        # ROOT - NOTHING
        if len(a) < self.J.shape[1]:  # nothing in root
            # print('root {} ; l={}'.format(a, depth))
            # not calculate here - root - no need
            rss_here = rss
            theta_here = theta
            inversion_here = inversion
            pass

        # FIRST EDGE - CALCULATE FIRST TINE
        elif len(a) == self.J.shape[1]:  # first calculation at depth 1

            theta_here, inversion_here, rss_here = self.calculate_theta_inversion_with_rss(self.J[a])
            #print(rss_here)

        # EDGE - UPDATE
        else:
            # print('a {} ; l={}'.format(a, depth))
            # Update RSS
            gama_insert, i_m_i, yi_xi_theta = self.calculate_gama_insert(a[-1], inversion, theta)
            xi = self.J[a[-1], 1:]
            #gama_insert = (yi_xi_theta ** 2) / (1 + i_m_i)
            rss_here = rss + gama_insert

            if rss_here >= self.bab_rss_min: # todo - just like that ???
                #print('cutting fucking branch ')
                return

            # Theta plus
            w = -1 / (1 + i_m_i)
            u = np.dot(inversion, xi.T)
            theta_here = theta + (-1 * yi_xi_theta * w * u)  # todo OK !!!! (changed [* -1] )

            # Inversion plus
            inversion_here = inversion + w * np.dot(u, u.T)

            # todo - cutting criterion

        # we go here either in roots , first edges , or regular edges
        aa = a.copy()
        bb = b.copy()
        while len(bb) > 0:
            if len(aa) + len(bb) < self._h_size:  # not enough to produce h subset in ancestor
                break

            # add from B to A
            aa.append(bb[0])
            # .. remove from B
            del bb[0]

            # Go down
            self.traverse_recursive(aa, bb, depth + 1, rss_here, theta_here, inversion_here)

            # remove from A
            del aa[-1]

        return
