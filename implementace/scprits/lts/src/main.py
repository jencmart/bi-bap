import numpy as np
import matplotlib.pyplot as plt
import math
import time


# currently support for np.ndarray and matrix
def validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold, use_intercept):
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


def preform_c_steps(theta_old, data, use_sum, sum_old, h_size, max_steps, threshold):  # vola se 10x

    for i in range(max_steps):  # vola se 50 x
        theta_new, h_new = c_step(theta_old, data, h_size)  # chyba tady

        if use_sum:
            sum_new = RSS(data[h_new, :], theta_new)
            if math.isclose(sum_old, sum_new, rel_tol=threshold):
                break
            sum_old = sum_new

        theta_old = theta_new

    if not use_sum:
        sum_new = RSS(data[h_new, :], theta_new)

    return theta_new, h_new, sum_new[0,0], i


def c_step(theta_old, data, h_size):
    abs_residuals = ABS_DIST(data, theta_old)  # chyba tady
    h_new = K_SMALLEST_SET(abs_residuals, h_size)
    theta_new = OLS(data[h_new, :])
    return theta_new, h_new


# for storage only
class SelectiveIterationResults:
    def __init__(self, arr_theta_hat, arr_h_index, arr_rss):
        self.arr_theta_hat = arr_theta_hat
        self.arr_h_index = arr_h_index
        self.arr_rss = arr_rss


class FastLtsRegression:
    def __init__(self):
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

    def fit(self, X, y,
            num_starts: 'number of initial starts (H1)' = 500,
            num_start_c_steps: 'number of initial C steps' = 2,
            num_starts_to_finish: 'number of H3 which`ll to finish' = 10,
            max_c_steps: 'self explanatory' = 50,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True,
            threshold: 'stopping criterion for Qold Qnew sum residuals in c-steps' = 1e-6):

        # Init some properties
        self._data = validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold, use_intercept)
        self._p = self._data.shape[1] - 1
        self._N = self._data.shape[0]


        if h_size == 'default':
            self._h_size = math.ceil((self._N + self._p + 1) / 2)  # todo with or without intercept?
        else:
            self._h_size = h_size

        # Selective iteration := h1 + few c-steps + find few with best rss
        # results = self.selective_iteration(num_starts, num_start_c_steps)  # todo time:
        # arr_best_idx = K_SMALLEST_SET(results.arr_rss, num_starts_to_finish)  # todo time: O(N)
        subset_results = self.create_all_h1_subsets(num_starts) # array of 500 Results (h1, thetha, inf)

        self.iterate_c_steps(subset_results, range(num_starts), False, num_start_c_steps, 0) # few c steps on all 500 results, all happens inplace
        SORT_K_SMALLEST_SET_INPLACE(subset_results, num_starts_to_finish) # arr results && indexes are sorted (sort first 10 from 500...)


        # C-steps till convergence, store number of iterations
        self.iterate_c_steps(subset_results, range(num_starts_to_finish), True, max_c_steps, threshold)

        final_rss = math.inf
        smallest_idx = 0
        for i in range(num_starts_to_finish):
            if subset_results[i].rss < final_rss:
                smallest_idx = i

        # final_results.arr_thetha_hat.shape (10,p,1) ndarray
        # final_results.arr_rss.shape (10,1,1) ndarray
        # final_results.arr_h_index.shape (10, 502) ndarray

        print(subset_results[smallest_idx].theta)
        print(subset_results[smallest_idx].h_subset)
        print(subset_results[smallest_idx].rss)
        print(subset_results[smallest_idx].n_iter)

        # ... Store results
        # theta_final = final_results.arr_theta_hat[arr_final_idx[0]]
        #
        # if use_intercept:
        #     self.intercept_ = theta_final[-1,0]  # last row last col
        #     self.coef_ = theta_final[:-1,0]  # for all but last column,  only first col
        # else:
        #     self.intercept_ = 0.0
        #     self.coef_ = theta_final[:,0]  # all rows, only first col
        #
        # self.h_subset_ = final_results.arr_h_index[arr_final_idx[0]].astype(int) # 0 0 protoze vybirame z (10,502)
        # self.rss_ = final_results.arr_rss[arr_final_idx[0]][0,0] # 0 0 protoze vybirame z (10,1,1)

        # h_subset_ ndarray (502,)
        # rss float32
        # coef ndarray (p,)
        # intercept float32

    # Select initial H1
    # ONLY ONE H1 ( one array of indexes to _data)
    def generate_h1_subset(self):
        if self._p >= self._N:
            J = self._data
        else:
            # create random permutation
            idx_all = np.random.permutation(self._N)
            # cut first p indexes and save the rest
            idx_initial = idx_all[:self._p]
            idx_rest = idx_all[self._p:]

            # create initial matrix of shape (p,p)
            J = self._data[idx_initial, :]

            # J[:,1:] == only X, without first y column
            rank = np.linalg.matrix_rank(J[:, 1:])

            while rank < self._p and J.shape[0] < self._N:
                # get first index from rest of the indexes
                current = idx_rest[[0],]
                idx_rest = idx_rest[1:, ]

                # add row on this index -fixed, ok
                J = np.append(J, self._data[current, :], axis=0)

                # and recalculate rank
                rank = np.linalg.matrix_rank(J[:, 1:])

        # OLS on J
        theta_zero_hat = OLS(J)

        # abs dist on N, and return h smallest
        abs_residuals = ABS_DIST(self._data, theta_zero_hat)
        indexes = K_SMALLEST_SET(abs_residuals, self._h_size) # vraci pole indexu, mohlo by vracet o theta
        return indexes

    class Storage:
        def __init__(self, a, b):
            self.data = []
            self.a = a
            self.b = b

        def update(self, row):
            for r in row:
                self.data.append(r)

        def finalize(self):
            return np.reshape(self.data, newshape=(self.a, self.b))

    class BetterResults:
        def __init__(self, h_subset, theta, rss, n_iter):
            self.theta = theta
            self.h_subset = h_subset
            self.rss = rss
            self.n_iter = n_iter

    def create_all_h1_subsets(self, num_starts):
        arr_results = []
        for i in range(num_starts):
            init_h1 = self.generate_h1_subset() # one array of indexes to h1
            arr_results.append(self.BetterResults(init_h1, OLS(self._data[init_h1, :]), math.inf, 0))
        return arr_results

    def iterate_c_steps(self, results, indexes, stop_on_rss, cnt_steps, threshold):

        for i in indexes: # bude brat v potaz jenom prvnich X
            theta, h_subset, rss, n_iter = preform_c_steps(results[i].theta, self._data, stop_on_rss, results[i].rss, self._h_size, cnt_steps, threshold)
            results[i].theta = theta
            results[i].h_subset = h_subset
            results[i].rss = rss
            results[i].n_iter += n_iter

    def selective_iteration(self, num_starts, num_start_c_steps):
        start_time = time.time()

        arr_theta_hat = []
        arr_h_subset = []
        arr_rss = []

        # for number of starts
        for i in range(num_starts):  # todo time O(num_starts * (n-p) * matrixSVD(pxp))
            # select initial h1
            init_h1 = self.generate_h1_subset()  # todo time: (n-p) * matrixSVD(pxp) CANT GO BETTER ?? [BINARY -- ?]
            # make few c-steps
            # (p,1)  (h_size,) (1,1)
            thetha_old = OLS(self._data[init_h1, :])

            theta, h_subset, rss, _ = preform_c_steps(thetha_old, self._data, False, None, self._h_size, num_start_c_steps, 0)
            print('theta shape', theta.shape)
            print('h     shape', h_subset.shape)
            print('rss   shape', rss.shape)

            print('theta tpye',type(theta))
            print('h     tpye',type(h_subset))
            print('rss   tpye',type(rss))
            exit(1)
            arr_theta_hat.append(theta)
            arr_h_subset.append(h_subset)
            arr_rss.append(rss.A1)

        arr_theta_hat = np.asarray(arr_theta_hat)
        arr_h_subset = np.asarray(arr_h_subset)
        arr_rss = np.asarray(arr_rss)
        elapsed_time = time.time() - start_time
        print('selective iteration', elapsed_time)  # 70 - 80

        return SelectiveIterationResults(arr_theta_hat, arr_h_subset, arr_rss)

    def iterate_till_convergence(self, results, arr_best_idx, max_c_steps, threshold):
        start_time = time.time()

        arr_theta_hat = []
        arr_h_subset = []
        arr_rss = []

        n_iter = 0
        for i in arr_best_idx:
            theta, h_subset, rss, n_iter = preform_c_steps(results.arr_theta_hat[i],
                                                           self._data,
                                                           True,
                                                           results.arr_rss[i],
                                                           self._h_size,
                                                           max_c_steps, threshold)

            arr_theta_hat.append(theta)
            arr_h_subset.append(h_subset)
            arr_rss.append(rss)

        arr_theta_hat = np.asarray(arr_theta_hat)
        arr_h_subset = np.asarray(arr_h_subset)
        arr_rss = np.asarray(arr_rss)
        elapsed_time = time.time() - start_time
        print('iterate till convergence', elapsed_time)  #

        return SelectiveIterationResults(arr_theta_hat, arr_h_subset, arr_rss), n_iter


##################
# MAIN FUNCTIONS #
##################
def RSS(input_data, theta):
    y = input_data[:, [0]]
    x = input_data[:, 1:]
    return (y - x * theta).T * (y - x * theta)


def OLS(input_data):
    # [0] .. diky tomu bude mit spravny shape
    y = input_data[:, [0]]
    x = input_data[:, 1:]
    return (x.T * x).I * x.T * y  # including intercept (last)


def ABS_DIST(data, theta):
    y = data[:, [0]]
    x = data[:, 1:]
    # Y (p+,1)
    # thetha (p+ , 1)
    # xx (n, p)
    return np.absolute(y - x * theta)


# what if I sort it...
# trying to think out of the box
def SORT_K_SMALLEST_SET_INPLACE(results, k_smallest):

    def kth_smallest(arr_results, left, right, k):
        # partition
        pivot = arr_results[right].rss
        pos = left
        for j in range(left, right):
            if arr_results[j].rss <= pivot:
                arr_results[pos], arr_results[j] = arr_results[j], arr_results[pos]  # swap whole results
                #indexes[pos], indexes[j] = indexes[j], indexes[pos]  # swap indexes also
                pos += 1

        arr_results[pos], arr_results[right] = arr_results[right], arr_results[pos]
        #indexes[pos], indexes[right] = indexes[right], indexes[pos]

        # finish
        if pos - left == k - 1:
            #return arr_results[:pos + 1], indexes[:pos + 1]  # values, indexes
            return
        # left part
        elif pos - left > k - 1:
            return kth_smallest(arr_results, left, pos - 1, k)
            # right part
        return kth_smallest(arr_results, pos + 1, right, k - pos + left - 1)

    kth_smallest(results, 0, len(results) - 1, k_smallest)
    return


def K_SMALLEST_SET(absolute_dist_in, k_smallest):
    absolute_dist_copy = np.copy(absolute_dist_in)

    indexes = np.arange(absolute_dist_copy.shape[0])
    absolute_dist = np.ravel(absolute_dist_copy)

    def kthSmallest(arr, indexes, left, right, k):
        # partition
        pivot = arr[right]
        pos = left
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[pos], arr[j] = arr[j], arr[pos]  # swap
                indexes[pos], indexes[j] = indexes[j], indexes[pos]  # swap indexes
                pos += 1

        arr[pos], arr[right] = arr[right], arr[pos]
        indexes[pos], indexes[right] = indexes[right], indexes[pos]

        # finish
        if pos - left == k - 1:
            return arr[:pos + 1], indexes[:pos + 1]  # values, indexes

        # left part
        elif pos - left > k - 1:
            return kthSmallest(arr, indexes, left, pos - 1, k)
            # right part
        return kthSmallest(arr, indexes, pos + 1, right, k - pos + left - 1)

    result_values, result_indexes = kthSmallest(absolute_dist, indexes, 0, absolute_dist.shape[0] - 1, k_smallest)
    return result_indexes

if __name__ == '__main__':
    # LINEAR DATA
    # data generated same way as in Rousseeuw and Driessen 2000
    X_original = np.random.normal(loc=0, scale=10, size=800)  # var = 100
    e = np.random.normal(loc=0, scale=1, size=800)  # var = 1
    y_original = 1 + X_original + e

    # OUTLIERS
    # multivariate N(mean = location, covariance)
    # diagonalni 25 I
    outliers = np.random.multivariate_normal(mean=[50, 0],
                                             cov=[[25, 0], [0, 25]],
                                             size=200)

    # FINAL DATA
    X = np.concatenate((X_original, outliers.T[0]), axis=0)
    y = np.concatenate((y_original, outliers.T[1]), axis=0)

    lts = FastLtsRegression()
    lts.fit(X, y, use_intercept=True)

    # print('wights: ', lts.coef_)
    # print('intercept: ', lts.intercept_)
    # print('rss: ', lts.rss_)
    # print('iters(+2):', lts.n_iter_)  # final inters only...
    # arr_idx = lts.h_subset_
    #
    # # Plot data
    # y_used = y[arr_idx]
    # X_used = X[arr_idx]
    # # nifty trick
    # mask = np.ones(y.shape[0], np.bool)
    # mask[arr_idx] = 0
    # y_not_used = y[mask]
    # X_not_used = X[mask]
    #
    # # Pot itself
    # plt.figure(figsize=(12, 8))
    # plt.plot(X_not_used, y_not_used, 'b.')
    # plt.plot(X_used, y_used, 'r.')
    # plt.plot(X, lts.coef_ * X + lts.intercept_, '-')
    # plt.show()