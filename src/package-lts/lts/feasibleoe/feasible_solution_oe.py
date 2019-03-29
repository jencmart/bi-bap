from lts.feasible.helpers import AbstractRegression
from lts.feasible.helpers import validate
import numpy as np
import math
import time
# import cppimport.import_hook
# import lts.feasibleoe.cpp.feasible_solution as cpp_solution

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
    def go_through_all_pairs(self, J, M, R, rss, inversion, residuals_J, residuals_M):
        delta = 1
        i_to_swap = None
        j_to_swap = None
        i_to_swap_oe = None
        j_to_swap_oe = None
        delta_oe = 2
        # go through all combinations
        for i in range(J.shape[0]):
            for j in range(M.shape[0]):
                # . calculate deltaRSS
                tmp_delta = self.calculate_delta_rss(J, M, inversion, residuals_J, residuals_M, i, j)
                tmp_delta_2 = self.calculate_delta_rss_oe(J, M, R, rss, inversion, residuals_J, residuals_M, i, j)

               # print('rss1, {}  [= {} + {}]'.format(rss[0, 0] + tmp_delta,   rss[0, 0], tmp_delta))
                #print('rss2, {}  [= {} * {}]'.format(rss[0, 0] * tmp_delta_2, rss[0, 0], tmp_delta_2))

                #print('-----')
                # todo POROVNEJ DELTY !

                # if delta rss < bestDeltaRss
                if tmp_delta < 0 and tmp_delta < delta:
                    delta = tmp_delta
                    i_to_swap = i
                    j_to_swap = j

                if tmp_delta_2 < 1 and tmp_delta_2 > 0 and tmp_delta_2 < delta_oe: # vetsi nez nula musi byt vzdy, ne ?
                    delta_oe = tmp_delta_2
                    i_to_swap_oe = i
                    j_to_swap_oe = j

        #if i_to_swap is None and i_to_swap_oe is not None:
         #   raise EnvironmentError
        #if i_to_swap is not None:
         #   print('returning i[{}] j[{}]'.format(i_to_swap, j_to_swap))
          #  print('returning i[{}] j[{}]'.format(i_to_swap_oe, j_to_swap_oe))
        return i_to_swap, j_to_swap, delta, delta_oe

    # ###########################################################################
    # ############### REFINEMENT ################################################
    # ###########################################################################

    def refinement_process(self, J, M, idx_initial, idx_rest):
        steps = 0

        while True:
            # data for delata eqation
            y = J[:, [0]]
            x = J[:, 1:]

            q, r = linalg.qr(x)  # X = QR ; x.T x = R.T R ; (pozor: cholesky je L * L.T kde l = R.T)
            # Q ... n x n
            # R ... n x p
            #
            #  Q.T *  ( x * w - y ) ^ 2
            #  Q.T * Q * R * w - Q.T * y
            #  R * w - Q.T * y
            #  R * w = Q.T * y
            #  r je ale n x p ... a od p do n jenom 0 (respektive h x p )
            p = x.shape[1]

            r1 = r[:p, :]  # only first p rows
            # transpose first
            qt = q.T

            q1 = qt[:p, :]  # only first p rows


            # Solve the equation x w = c for x, assuming a is a triangular matrix
            theta = linalg.solve_triangular(r1, q1*y)  # p x substitution -- works as expected
            inversion = (x.T * x).I
            theta = inversion * x.T * y  # OLS

            residuals_J = y - x * theta
            residuals_M = (M[:, [0]]) - (M[:, 1:]) * theta
            # rss todo
            rss = residuals_J.T * residuals_J

            i_to_swap, j_to_swap, delta, delta_oe = self.go_through_all_pairs(J, M, r1, rss, inversion, residuals_J, residuals_M)
            if delta >= 0: # todo zmenit pak podminku na delta_oe < 1
                break

            else:  # swap i and j [TOGHETHER WITH INDEXES] ; je to ok - SWAPUJEME SPRAVNE

                # todo - update RSS (easy) ;
                rss = rss * delta_oe
                row_to_add = np.copy(M[j_to_swap, 1:])

                # qr_update copies it
                q_added, r_added = self.qr_update(q,r, row_to_add)

                q_added_correct, r_added_correct = linalg.qr_insert(q, r, row_to_add, q.shape[0])

                print('*********************************************')
                print('*********************************************')
                #print('------- R MATRIXES -----------------')
                #print(r_added)
                #print('------ and correct R')
                #print(r_added_correct)

                #print('---------Q MATRIXES ----------------')

                # print(np.allclose(np.dot(q_added.T, q_added), np.eye(q_added.shape[0])))
                print(q_added - q_added_correct) # lisi se jen znamenkama
                print('------ and correct Q')
                #print()
                print('*********************************************')
                print('*********************************************')

                #
                # todo - update X.T X -1 --- problem  ;
                # todo - s tim se poji - update Q and R
                #
                #  todo - update theta - easy kdyz mam xtx-1

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

    def qr_update(self, q, r, row):
        n = q.shape[0]  # rows
        p = r.shape[1]  # cols
        cnt_rows = 1

        shape = [n + cnt_rows,  n + cnt_rows]

        # create new matrix
        qnew = np.zeros(shape, dtype=float)
        shape[1] = p
        rnew = np.zeros(shape, dtype=float)

        # fill matrix r
        rnew[:n, :] = np.copy(r)
        rnew[n:, :] = row  # just one row in this case (1,7 into 1,6)

        # fill matrix q
        qnew[:-cnt_rows, :-cnt_rows] = q
        for j in range(n, n + cnt_rows): # again, neni treba loop; jen posledni radek
            qnew[j, j] = 1

        n = n + 1
        # rotate last row and update both matrix
        limit = min(n - 1, p) # opet, autimaticky je to p (maji n-1 kvuli fortranu??) ne kvuli poslednimu sloupku q ?

        for j in range(limit):  # 0, 1 ... p-1
            cos, sin, R = self.calculate_cos_sin(rnew[j,j], rnew[n-1,j]) # edge of triangle , last row

            rnew[j, j] = R # some hack as they have
            rnew[n - 1, j] = 0

            # rotate rnew
            num = p-j # mame stale jen p
            rowX = rnew[j,:]
            rowY = rnew[n-1,:]


            # blas srot
            for i in range(j+1, p): # vzdy od j do konce
                temp = cos * rowX[i] + sin * rowY[i]
                rnew[n-1, i] = cos * rowY[i] - sin * rowX[i] # Y
                rnew[j, i] = temp # X

            # rotate qnew
            qrowX = qnew[:, j] # j ty slouepk
            qrowY = qnew[:, n-1] # posledni slopek

            # blas srot
            for i in range(n): # vzdy od j do konce
                temp = cos * qrowX[i] + sin * qrowY[i]
                qnew[i, n-1] = cos * qrowY[i] - sin * qrowX[i]  # Y
                qnew[i, j] = temp  # X

        # permutate ???
        # nope

        return qnew, rnew

    # ###########################################################################
    # ############### DELTA RSS #################################################
    # ###########################################################################


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
            r = math.sqrt( f**2 + g**2)
            cos = f / r
            sin = g / r

        if abs(f) > abs(g) and cos < 0:
            cos = -cos
            sin = -sin
            r = -r

        return cos, sin, r

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

    def calculate_delta_rss_oe(self, J, M, R1, rss,  inversion, residuals_J, residuals_M, i, j):
        # x * (R.T * R ) ^-1 * x = v.T v
        # where
        # R.T * v.T = xi.T
            # pridani
        xi = M[j, 1:]
        vi = linalg.solve_triangular(R1.T, xi.T, lower=True)
        #print('vi')
        #print(vi.shape)
        #print(type(vi.T))
        i_m_i = np.dot(vi.T, vi)  #vi.T * vi # dot product PRIDANI !!!
        i_m_i = i_m_i[0, 0]
            # odebrani
        xj = J[i, 1:]
        vj = linalg.solve_triangular(R1.T, xj.T, lower=True)
        j_m_j = np.dot(vj.T, vj)
        j_m_j = j_m_j[0, 0]

        # oboje (odebrani rtr pridani)
        # xj * (R.T * R ) ^-1 * xi = xj * u.T
        #  R * u.T = v kde v je resenim R.T * v.T = xi.T (teda vi ??? )
        u = linalg.solve_triangular(R1, vi)
        i_m_j = np.dot(xj, u)  # xj * u  # check if not xi
        i_m_j = i_m_j[0, 0]

        ei = residuals_M[j, 0] # ano, opravdu opacne
        ej = residuals_J[i, 0]

        # print('************')
        # print('IMI')
        # print(i_m_i)
        # print('JMJ')
        # print(j_m_j)
        # print('IMJ')
        # print(i_m_j)
        # print(ei)
        # print(ej)
        # print('************')
        #print('rss')
        #rss = rss[0,0]
        #print('rss---')
        nom = (1 + i_m_i + 1/rss * ei*ei) * (1 - j_m_j - 1/rss * ej*ej) + (i_m_j + 1/rss * ei*ej) * (i_m_j + 1/rss * ei*ej)
        denom = (1 + i_m_i - j_m_j + i_m_j * i_m_j - i_m_i * j_m_j)
        ro = nom / denom
        #print(ro)
        return ro
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
