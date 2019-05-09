import numpy as np
import math
import time
import cppimport.import_hook
import lts.fastlts.cpp.fastlts as cpp_solution


###########################
# To work way you expect, you must adhere the naming convention right:
# xxx.cpp
# PYBIND11_MODULE(xxx, m)
# something.py
# my_import =  cppimport.imp("xxx")
###########################
# eigen_lts = cppimport.imp("../src/fastlts")


class LTSRegressorFastCPP:
    def __init__(self,
                 num_starts: 'number of initial starts (H1)' = 500,
                 num_initial_c_steps: 'number of initial C steps' = 2,
                 num_starts_to_finish: 'number of H3 which`ll to finish' = 10,
                 max_steps: 'self explanatory' = 50,
                 threshold: 'stopping criterion Qold Qnew' = 1e-6,
                 use_intercept=True):

        # number of initial starts starts
        self._num_starts = num_starts

        # maximum number of iterations
        self._max_steps = max_steps

        # set using intercept
        self._use_intercept = use_intercept

        # number of initial steps
        self._num_initial_c_steps = num_initial_c_steps

        # num steps to finish
        self._num_starts_to_finish = num_starts_to_finish

        # threshold
        self._threshold = threshold

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
    @staticmethod
    def _validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold,
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

            h_size: 'default := (n + p + 1) / 2' = 'default'):

        X, y = self._validate(X, y, h_size, self._num_initial_c_steps, self._num_starts_to_finish,
                              self._max_steps, self._threshold,
                              self._use_intercept)

        p = X.shape[1]
        n = X.shape[0]
        _h_size = calculate_h_size(n, p, h_size)

        # _h_size = math.ceil((X.shape[0] + X.shape[1] + 1) / 2) if h_size == 'default' else h_size  # N + p + 1
        # print('fast-lts : {}'.format(_h_size))
        eigen_result = cpp_solution.fast_lts(X, y, self._num_starts, self._num_initial_c_steps,
                                             self._num_starts_to_finish,
                                             _h_size,
                                             self._max_steps, self._threshold)

        # ... Store best result
        weights = eigen_result.get_theta()
        if self._use_intercept:
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


class LTSRegressorFast:
    def __init__(self,
                 num_starts: 'number of initial starts (H1)' = 500,
                 num_initial_c_steps: 'number of initial C steps' = 2,
                 num_starts_to_finish: 'number of H3 which`ll to finish' = 10,
                 max_steps: 'self explanatory' = 50,
                 threshold: 'stopping criterion Qold Qnew' = 1e-6,
                 use_intercept=True):

        # number of initial starts starts
        self._num_starts = num_starts

        # maximum number of iterations
        self._max_steps = max_steps

        # set using intercept
        self._use_intercept = use_intercept

        # number of initial steps
        self._num_initial_c_steps = num_initial_c_steps

        # num steps to finish
        self._num_starts_to_finish = num_starts_to_finish

        # threshold
        self._threshold = threshold

        # public
        self.n_iter_ = None
        self.coef_ = None
        self.intercept_ = None
        self.h_subset_ = None
        self.rss_ = None
        # process time - for benchmark only
        self.time1_ = None
        self.time2_ = None
        self.time3_ = None
        self.time_total_ = None

    # currently support for np.ndarray and matrix
    @staticmethod
    def _validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold, use_intercept):
        if X is None or not isinstance(X, (np.ndarray, np.matrix)):
            raise Exception('X must be  type array or np.ndarray or np.matrix')
        if y is None or not isinstance(y, (np.ndarray, np.matrix)):
            raise Exception('y must be  type array or np.ndarray or np.matrix')

        if X.ndim == 1:
            X = np.reshape(X, [X.shape[0], 1])
        if y.ndim == 1:
            y = np.reshape(y, [y.shape[0], 1])

        if type(X) is not np.matrix:
            X = np.asmatrix(X)
        if type(y) is not np.matrix:
            y = np.asmatrix(y)

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
            merged = np.concatenate([y, X, np.ones((X.shape[0], 1))], axis=1)
        else:
            merged = np.concatenate([y, X], axis=1)

        return np.asmatrix(merged)

    def fit(self, X, y,
            h_size: 'default := (n + p + 1) / 2' = 'default'):

        # Init some properties
        data = self._validate(X, y, h_size, self._num_initial_c_steps, self._num_starts_to_finish,
                              self._max_steps,
                              self._threshold,
                              self._use_intercept)

        # h size
        p = data.shape[1] - 1
        n = data.shape[0]
        _h_size = calculate_h_size(n, p, h_size)

        # IF N > 1500
        # 1. CREATE 5 SUBSETS OF THE DATA
        # ON EACH SUBSET CREATE SUBSET RESULTS ( NUM STARTS / 5 )
        # WHAT SHOULD BE THE SIZE OF H -- how many vectors to choose from
        #       for example n+p/2 ... 750 from 300 is not acceptable
        # they say : hsub = [nsub(h/n)] nsub = 300 n = 1500
        # I SUPPOSE (and it make sense)
        # nested extension : DATA = TEN SUBSET (ex. 300) H_SIZE := subset_size * h/n

        # on each subset carry out few c steps
        # and from each subset select 10 best results

        # merge all best results together --> 50 results
        # carry out 2 c steps
        # select 10 best
        # iterate till convergence

        # Selective iteration := h1 + few c-steps + find few with best rss
        # result = eigen_lts.fast_lts(data, num_starts,
        # num_start_c_steps, num_starts_to_finish, max_c_steps, h_size, threshold)
        # X = data[:, 1:]
        # y = data[:, :1]

        time1 = time.process_time()
        subset_results = self.create_all_h1_subsets(self._num_starts, _h_size, data)  # array of 500 Results (h1, thetha, inf)
        self.time1_ = time.process_time() - time1

        time2 = time.process_time()
        self.iterate_c_steps(data, _h_size, subset_results, self._num_starts, False, self._num_initial_c_steps,
                             0)  # few c steps on all 500 results, all happens inplace
        k_smallest_inplace(subset_results,
                           self._num_starts_to_finish)  # arr results && indexes are sorted (sort first 10 from 500...)
        self.time2_ = time.process_time() - time2

        # C-steps till convergence
        time3 = time.process_time()
        self.iterate_c_steps(data, _h_size, subset_results, self._num_starts_to_finish, True, self._max_steps,
                             self._threshold)
        # select the best one
        best_result = subset_results[0]
        for i in range(self._num_starts_to_finish):
            best_result = subset_results[i] if subset_results[i].rss < best_result.rss else best_result
        self.time3_ = time.process_time() - time3
        self.time_total_ = self.time1_ + self.time2_ + self.time3_

        # ... Store best result
        if self._use_intercept:
            self.intercept_ = best_result.theta[-1, 0]  # last row first col
            self.coef_ = np.ravel(best_result.theta[:-1, 0])  # for all but last column,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = np.ravel(best_result.theta[:, 0])  # all rows, only first col

        self.h_subset_ = best_result.h_subset.astype(int)
        self.rss_ = best_result.rss
        self.n_iter_ = best_result.n_iter

    # Select initial H1
    # ONLY ONE H1 ( one array of indexes to data)
    @staticmethod
    def generate_h1_subset(_h_size, data):
        p = data.shape[1] - 1
        N = data.shape[0]

        if p >= N:
            J = data
        else:
            # create random permutation
            idx_all = np.random.permutation(N)
            # cut first p indexes and save the rest
            idx_initial = idx_all[:p]
            idx_rest = idx_all[p:]

            # create initial matrix of shape (p,p)
            J = data[idx_initial, :]

            # J[:,1:] == only X, without first y column
            rank = np.linalg.matrix_rank(J[:, 1:])

            while rank < p and J.shape[0] < N:
                # get first index from rest of the indexes
                current = idx_rest[[0], ]
                idx_rest = idx_rest[1:, ]

                # add row on this index -fixed, ok
                J = np.append(J, data[current, :], axis=0)

                # and recalculate rank
                rank = np.linalg.matrix_rank(J[:, 1:])

        # OLS on J
        theta_zero_hat = ols(J)

        # abs dist on N, and return h smallest
        abs_residuals = abs_dist(data, theta_zero_hat)
        indexes = k_smallest(abs_residuals, _h_size)
        return indexes

    class ResultPython:
        def __init__(self, h_subset, theta, rss_, n_iter):
            self.h_subset = h_subset  # array
            self.theta = theta  # matrix
            self.rss = rss_  # double
            self.n_iter = n_iter  # integer

    def create_all_h1_subsets(self, num_starts, _h_size, data):
        arr_results = []
        for i in range(num_starts):
            init_h1 = self.generate_h1_subset(_h_size, data)  # one array of indexes to h1
            arr_results.append(self.ResultPython(init_h1, ols(data[init_h1, :]), math.inf, 0))
        return arr_results

    def iterate_c_steps(self, data, _h_size, results, length, stop_on_rss, cnt_steps, threshold):

        for i in range(length):  # only first X
            theta, h_subset, rss_, n_iter = self._preform_c_steps(results[i].theta, data, stop_on_rss, results[i].rss,
                                                                  _h_size, cnt_steps, threshold)
            results[i].theta = theta
            results[i].h_subset = h_subset
            results[i].rss = rss_
            results[i].n_iter += n_iter

    @staticmethod
    def _preform_c_steps(theta_old, data, use_sum, sum_old, h_size, max_steps, threshold):  # vola se 10x

        if max_steps == 0:
            exit(10)

        j = 0
        for i in range(max_steps):
            # c step
            abs_residuals = abs_dist(data, theta_old)  # nested extension
            h_new = k_smallest(abs_residuals, h_size)  #
            theta_new = ols(data[h_new, :])
            # ! c step

            if use_sum:
                sum_new = rss(data[h_new, :], theta_new)
                if math.isclose(sum_old, sum_new, rel_tol=threshold):
                    j = i + 1  # include last step
                    break
                sum_old = sum_new
            theta_old = theta_new

        if not use_sum:
            sum_new = rss(data[h_new, :], theta_new)

        if j == 0:
            j = max_steps

        return theta_new, h_new, sum_new[0, 0], j


def calculate_h_size(n, p, h_size):
    if h_size == 'default':
        s = math.floor((n + p + 1) / 2)  # greatest integer function ~ floor
    else:
        s = h_size

    return s

##################
# MAIN FUNCTIONS #
##################
def rss(input_data, theta):
    y = input_data[:, [0]]
    x = input_data[:, 1:]
    return (y - x * theta).T * (y - x * theta)


def ols(input_data):
    # [0] .. diky tomu bude mit spravny shape
    y = input_data[:, [0]]
    x = input_data[:, 1:]
    return (x.T * x).I * x.T * y  # including intercept (last)


def abs_dist(data, theta):
    # Y (p+,1)
    # theta (p+ , 1)
    # xx (n, p)
    return np.absolute(data[:, [0]] - data[:, 1:] * theta)


def k_smallest_inplace(results, kth):
    def kth_smallest(arr_results, left, right, k):
        # partition
        pivot = arr_results[right].rss
        pos = left
        for j in range(left, right):
            if arr_results[j].rss <= pivot:
                arr_results[pos], arr_results[j] = arr_results[j], arr_results[pos]  # swap whole results
                pos += 1

        arr_results[pos], arr_results[right] = arr_results[right], arr_results[pos]

        # finish
        if pos - left == k - 1:
            # return arr_results[:pos + 1], indexes[:pos + 1]  # values, indexes
            return
        # left part
        elif pos - left > k - 1:
            return kth_smallest(arr_results, left, pos - 1, k)
            # right part
        return kth_smallest(arr_results, pos + 1, right, k - pos + left - 1)

    kth_smallest(results, 0, len(results) - 1, kth)
    return


def k_smallest(absolute_dist_in, kth_smallest):
    absolute_dist_copy = np.copy(absolute_dist_in)

    indexes = np.arange(absolute_dist_copy.shape[0])
    absolute_dist = np.ravel(absolute_dist_copy)

    def kth_smallest2(arr, idx_arr, left, right, k):
        # partition
        pivot = arr[right]
        pos = left
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[pos], arr[j] = arr[j], arr[pos]  # swap
                idx_arr[pos], idx_arr[j] = idx_arr[j], idx_arr[pos]  # swap indexes
                pos += 1

        arr[pos], arr[right] = arr[right], arr[pos]
        idx_arr[pos], idx_arr[right] = idx_arr[right], idx_arr[pos]

        # finish
        if pos - left == k - 1:
            return arr[:pos + 1], idx_arr[:pos + 1]  # values, indexes

        # left part
        elif pos - left > k - 1:
            return kth_smallest2(arr, idx_arr, left, pos - 1, k)
            # right part
        return kth_smallest2(arr, idx_arr, pos + 1, right, k - pos + left - 1)

    result_values, result_indexes = kth_smallest2(absolute_dist, indexes, 0, absolute_dist.shape[0] - 1, kth_smallest)
    return result_indexes
