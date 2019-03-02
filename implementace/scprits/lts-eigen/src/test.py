import cppimport
import numpy as np
import math
import time

eigen_lts = cppimport.imp("somecode")



class FastLtsRegression:
    def __init__(self):
        # public
        self.n_iter_ = None
        self.coef_ = None
        self.intercept_ = None
        self.h_subset_ = None
        self.rss_ = None

    # currently support for np.ndarray and matrix
    def _validate(self, X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold, use_intercept):
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
            num_starts: 'number of initial starts (H1)' = 500,
            num_start_c_steps: 'number of initial C steps' = 2,
            num_starts_to_finish: 'number of H3 which`ll to finish' = 10,
            max_c_steps: 'self explanatory' = 50,
            h_size: 'default := (n + p + 1) / 2' = 'default',
            use_intercept=True,
            threshold: 'stopping criterion for Qold Qnew sum residuals in c-steps' = 1e-6):

        # Init some properties
        data = self._validate(X, y, h_size, num_start_c_steps, num_starts_to_finish, max_c_steps, threshold, use_intercept)

        _h_size = math.ceil((data.shape[0] + data.shape[1] ) / 2) if h_size == 'default' else h_size # N + (p-1) + 1

        # IF N > 1500
        # 1. CREATE 5 SUBSETS OF THE DATA
        # ON EACH SUBSET CREATE SUBSET RESULTS ( NUMSTARTS / 5 )
            # WHAT SHOULD BE THE SIZE OF H -- how many vectors to choose from ( for example n+p/2 ... 750 from 300 is not acceptable
            # they say : hsub = [nsub(h/n)] nsub = 300 n = 1500
            # I SUPPOSE (and it make sense)
                # nested extension : DATA = TEN SUBSET ( 300 napriklad..) H_SIZE := subset_size * h/n ???? jo dava smysl ...lece pres 50% opet..

        # on each subset carry out few c steps
        # and from each subset select 10 best results

        # merge all best results together --> 50 results
        # carry out 2 c steps
        # select 10 best
        # iterate till convergence

        # Selective iteration := h1 + few c-steps + find few with best rss
        # result = eigen_lts.fast_lts(data, num_starts, num_start_c_steps, num_starts_to_finish, max_c_steps, h_size, threshold)
        X = data[:, 1:]
        y = data[:, :1]

        print('start')
        eigen_result = eigen_lts.fast_lts(X, y, num_starts, num_start_c_steps, num_starts_to_finish, _h_size, max_c_steps, threshold)
        self.eigen_weights = eigen_result.get_theta()
        self.eigen_h_subset = eigen_result.get_h_subset()
        self.eigen_rss = eigen_result.get_rss()
        self.eigen_iters = eigen_result.get_n_inter()
        self.eigen_time1 = eigen_result.get_time_1()
        self.eigen_time2 = eigen_result.get_time_2()
        self.eigen_time3 = eigen_result.get_time_3()

        print('done')
        return;

        time1 = time.process_time()
        subset_results = self.create_all_h1_subsets(num_starts, _h_size, data) # array of 500 Results (h1, thetha, inf)
        self.time1 =  time.process_time() - time1


        time2 = time.process_time()
        self.iterate_c_steps(data, _h_size, subset_results, num_starts, False, num_start_c_steps, 0) # few c steps on all 500 results, all happens inplace
        k_smallest_inplace(subset_results, num_starts_to_finish) # arr results && indexes are sorted (sort first 10 from 500...)
        self.time2 = time.process_time() - time2

        # C-steps till convergence
        time3 = time.process_time()
        self.iterate_c_steps(data, _h_size, subset_results, num_starts_to_finish, True, max_c_steps, threshold)
        # select the best one
        best_result = subset_results[0]
        for i in range(num_starts_to_finish):
            best_result = subset_results[i] if subset_results[i].rss < best_result.rss else best_result
        self.time3 = time.process_time() - time3

        # ... Store best result
        if use_intercept:
            self.intercept_ = best_result.theta[-1,0]  # last row first col
            self.coef_ = np.ravel ( best_result.theta[:-1,0] ) # for all but last column,  only first col
        else:
            self.intercept_ = 0.0
            self.coef_ = np.ravel( best_result.theta[:,0] ) # all rows, only first col

        self.h_subset_ = best_result.h_subset.astype(int)
        self.rss_ = best_result.rss
        self.n_iter_ = best_result.n_iter



    # Select initial H1
    # ONLY ONE H1 ( one array of indexes to data)
    def generate_h1_subset(self, _h_size, data):
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
                current = idx_rest[[0],]
                idx_rest = idx_rest[1:, ]

                # add row on this index -fixed, ok
                J = np.append(J, data[current, :], axis=0)

                # and recalculate rank
                rank = np.linalg.matrix_rank(J[:, 1:])

        # OLS on J
        theta_zero_hat = ols(J)

        # abs dist on N, and return h smallest
        abs_residuals = abs_dist(data, theta_zero_hat)
        indexes = k_smallest(abs_residuals, _h_size) # vraci pole indexu, mohlo by vracet o theta
        return indexes

    class BetterResults:
        def __init__(self, h_subset, theta, rss, n_iter):
            self.h_subset = h_subset  # array
            self.theta = theta # matrix
            self.rss = rss # double
            self.n_iter = n_iter # integer


    def create_all_h1_subsets(self, num_starts, _h_size, data):
        arr_results = []
        for i in range(num_starts):
            init_h1 = self.generate_h1_subset(_h_size, data) # one array of indexes to h1
            arr_results.append(self.BetterResults(init_h1, ols(data[init_h1, :]), math.inf, 0))
        return arr_results

    def iterate_c_steps(self, data, _h_size, results, length, stop_on_rss, cnt_steps, threshold):

        for i in range(length): # bude brat v potaz jenom prvnich X
            theta, h_subset, rss, n_iter = self._preform_c_steps(results[i].theta, data, stop_on_rss, results[i].rss, _h_size, cnt_steps, threshold)
            results[i].theta = theta
            results[i].h_subset = h_subset
            results[i].rss = rss
            results[i].n_iter += n_iter


    def _preform_c_steps(self, theta_old, data, use_sum, sum_old, h_size, max_steps, threshold):  # vola se 10x

        for i in range(max_steps):
            # c step
            abs_residuals = abs_dist(data, theta_old)  # nested extension : DATA = TEN SUBSET ( 300 napriklad..) H_SIZE := subset_size * h/n ???? jo dava smysl ...lece pres 50% opet..
            h_new = k_smallest(abs_residuals, h_size)  #
            theta_new = ols(data[h_new, :])
            # ! c step

            if use_sum:
                sum_new = rss(data[h_new, :], theta_new)
                if math.isclose(sum_old, sum_new, rel_tol=threshold):
                    break
                sum_old = sum_new
            theta_old = theta_new

        if not use_sum:
            sum_new = rss(data[h_new, :], theta_new)

        return theta_new, h_new, sum_new[0, 0], i


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



# what if I sort it...
# trying to think out of the box
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
            #return arr_results[:pos + 1], indexes[:pos + 1]  # values, indexes
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

def generate_data(cnt, outlier_percentage=25):
    # LINEAR DATA
    # data generated same way as in Rousseeuw and Driessen 2000
    N_clear = cnt - int(math.floor(cnt/100*outlier_percentage))
    N_dirty = int(math.ceil(cnt/100*outlier_percentage))

    X_original = np.random.normal(loc=0, scale=10, size=N_clear)  # var = 100
    e = np.random.normal(loc=0, scale=1, size=N_clear)  # var = 1
    y_original = 1 + X_original + e
    # OUTLIERS
    # multivariate N(mean = location, covariance)
    # diagonalni 25 I
    outliers = np.random.multivariate_normal(mean=[50, 0],
                                             cov=[[25, 0], [0, 25]],
                                             size=N_dirty)

    # FINAL DATA
    X = np.concatenate((X_original, outliers.T[0]), axis=0)
    y = np.concatenate((y_original, outliers.T[1]), axis=0)

    return X,y

if __name__ == '__main__':

    X, y = generate_data(1000000, outlier_percentage=40)
    lts = FastLtsRegression()
    lts.fit(X, y, use_intercept=True)
    print('\n')
    # print('wights: ', lts.coef_)
    # print('intercept: ', lts.intercept_)
    # print('rss: ', lts.rss_)
    # print('iters:', lts.n_iter_)  # final inters only...
    # print('t1: ', lts.time1)
    # print('t2: ', lts.time2)
    # print('t3: ', lts.time3)
    # print('total cpu time: ', lts.time1 + lts.time2 + lts.time2 + lts.time3)
    # print('****************\n')

    print('c code')
    print('weights: ', lts.eigen_weights)
    print('rss: ', lts.eigen_rss)
    print('iter: ', lts.eigen_iters)
    print('t1: ', lts.eigen_time1)
    print('t2: ', lts.eigen_time2)
    print('t3: ', lts.eigen_time3)
    print('total cpu time: ', lts.eigen_time1 + lts.eigen_time2 + lts.eigen_time3)

    # x = np.arange(12)
    # print(x)
    # print(code.add_arrays(x, x))
    # A = np.array([[1, 2, 1],
    #               [2, 1, 0],
    #               [-1, 1, 2]])
    # print('NP ARRAY', A)
    # print('INVERZE', code.inv(A))
    # print('DETERMINANT', code.det(A))
    # print('INVERZE', code.inv(A))


