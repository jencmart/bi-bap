import lts.feasible.feasible_solution as feasible
from data.data_generator import generate_data_ND
import data.data_generator as generator
import lts.fastlts.fast_lts as fastlts
from scipy import spatial
import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression


def test_cpp(n=100, p=2, algorithm='fsa', calculation='inv', num_starts=10, max_steps=50, intercept=True, h_size='default'):
    print('test started...')
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = feasible.LTSRegressorFeasibleCPP(num_starts=num_starts, max_steps=max_steps, use_intercept=intercept,
                                           algorithm=algorithm, calculation=calculation)
    lts.fit(x, y, h_size=h_size)

    # lts
    weights_lts = lts.coef_

    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # OLS on the clean data
    lts.fit(x_clean, y_clean, h_size=x_clean.shape[0])
    weights_global_minimum = lts.coef_

    # cos similarity
    result = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)
    print('cos: ', result)
    print('...test finished')


def test_numpy(n=100, p=2, algorithm='fsa', calculation='inv'):
    print('test started...({} x {})'.format(n, p))
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = feasible.LTSRegressorFeasible(num_starts=10, max_steps=50, use_intercept=True,
                                        algorithm=algorithm, calculation=calculation)
    lts.fit(x, y, h_size='default')

    # lts
    weights_lts = lts.coef_

    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # OLS on the clean data
    lts.fit(x_clean, y_clean, h_size=x_clean.shape[0])
    weights_global_minimum = lts.coef_

    # cos similarity
    result = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)
    print('cos: ', result)
    print('...test finished')


def test_fast_feasible_numpy(n=100, p=2, algorithm='fsa', calculation='inv', use_intercept=True):
    print('test started...({} x {})'.format(n, p))

    # GENERATE THE DATA
    x, y, x_clean, y_clean = generator.generate_dataset_simple(n, p, outlier_ratio=0.3)

    multiple_starts_cnt = 100

    h_subsets = None

    # FIT THE FAST LTS

    if multiple_starts_cnt > 1:
        for i in range(multiple_starts_cnt):

            lts = fastlts.LTSRegressorFast(use_intercept=use_intercept, num_starts=500,
                                           num_initial_c_steps=2,
                                           max_steps=5)
            lts.fit(x, y, h_size='default')

            if h_subsets is None:
                sub = lts.h_subset_
                sub = np.reshape(sub, [1, sub.shape[0]])
                h_subsets = sub
            else:
                sub = lts.h_subset_
                sub = np.reshape(sub, [1, sub.shape[0]])
                h_subsets = np.concatenate((h_subsets, sub), axis=0)

        #  trim duplicated results
        h_subsets = np.unique(h_subsets, axis=0)
        print('FAST LTS (multiple)')
        print('--------------')

    else:
        lts = fastlts.LTSRegressorFast(use_intercept=use_intercept, num_starts=500, num_initial_c_steps=2, max_steps=500)
        lts.fit(x, y, h_size='default')
        h_subsets = lts.h_subset_
        lts.fit(x, y)
        h1 = lts.h_subset_
        w1 = lts.coef_
        print('FAST LTS:')
        # print data
        print('rss: ', lts.rss_)
        print('itr: ', lts.n_iter_)
        print('sec: ', lts.time_total_)
        print('--------------')

    # FIT FEASIBLE SOLUTION using subset from FAST LTS
    print('---------')
    lts = feasible.LTSRegressorFeasible(num_starts=10, max_steps=50, use_intercept=use_intercept,
                                        algorithm=algorithm, calculation=calculation)

    lts.fit(x, y, h_size='default', index_subset=h_subsets)
    h2 = lts.h_subset_
    w2 = lts.coef_
    # print data
    print('FAST->FEASIBLE (' + algorithm + '):')
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    print('--------------')

    # FIT FEASIBLE SOLUTION without subset
    print('---------')
    lts = feasible.LTSRegressorFeasible(num_starts=10, max_steps=500, use_intercept=use_intercept,
                                        algorithm=algorithm, calculation=calculation)

    lts.fit(x, y, h_size='default')
    htmp = lts.h_subset_
    wtmp = lts.coef_
    # print data
    print('FEASIBLE no subset (' + algorithm + '):')
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    print('--------------')


    # LEAST BUT NOT LAST FIT CLEAN DATA
    print('---------')
    lts = feasible.LTSRegressorFeasible(num_starts=10, max_steps=50, use_intercept=use_intercept,
                                        algorithm=algorithm, calculation=calculation)
    lts.fit(x_clean, y_clean, h_size='default')
    h3 = lts.h_subset_
    w3 = lts.coef_
    # print data
    print('CLEAN (' + algorithm + '):')
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    print('---------')

    cos_sim = 1 - spatial.distance.cosine(w2, w3)
    print('--- cos sim ---')
    print(cos_sim)


def test_fast_and_feasible(n=100, p=2):
    print('test started...')
    x, y, x_clean, y_clean = generate_data_ND(n, p)

    print(x.shape)
    lts = fastlts.LTSRegressorFastCPP()
    lts.fit(x, y, use_intercept=True, num_starts=10)
    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    h1 = lts.h_subset_
    h1.sort()
    h1 = np.asarray(h1)

    print('........................')

    lts = feasible.LTSRegressorFeasible(num_starts=10, max_steps=50, use_intercept=True,
                                        algorithm='moea', calculation='qr')
    lts.fit(x, y, h_size='default', index_subset=[h1])
    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    h2 = lts.h_subset_
    h2.sort()

    print('........................')

    mask = np.ones(x.shape[0], np.bool)
    mask[h2] = 0
    all_idx = np.arange(x.shape[0])
    idx_ones = all_idx[h2]
    idx_zeroes = all_idx[mask]
    changed_idx = np.concatenate((idx_ones, idx_zeroes), axis=0)

    x_first = x[idx_ones]
    x_rest = x[idx_zeroes]
    x = np.concatenate((x_first, x_rest), axis=0)

    y_first = y[idx_ones]
    y_rest = y[idx_zeroes]
    y = np.concatenate((y_first, y_rest), axis=0)

    lts.fit_exact(x, y, algorithm='bab', use_intercept=True)
    # print data
    print('rss: ', lts.rss_)
    print('cut: ', lts.n_iter_)
    print('sec: ', lts.time_total_)
    h3 = lts.h_subset_
    h3.sort()
    true_idx = changed_idx[h3]
    true_idx.sort()

    print('........................')
    print(h1)
    print(h2)
    print(true_idx)

def solve_clean(X_clean, y_clean, h_size, intercept):
    X_clean, y_clean = validate(X_clean, y_clean, use_intercept=intercept)
    X_clean = np.asmatrix(X_clean)
    y_clean = np.asmatrix(y_clean)
    q, r = linalg.qr(X_clean)
    p = r.shape[1]
    r1 = r[:p, :]
    qt = q.T
    q1 = qt[:p, :]
    theta = linalg.solve_triangular(r1, q1 * y_clean)  # p x substitution
    residuals = y_clean - X_clean * theta

    # h subset
    residuals = residuals.T
    residuals = np.ravel(residuals)
    sort_args = np.argsort(residuals)
    h_subset = sort_args[:h_size]

    # rss on h subset
    residuals.sort()
    residuals_h = residuals[:h_size]
    rss_clean = np.dot(residuals_h, residuals_h.T)

    if intercept:
        intercept = theta[-1, 0]  # last row first col
        coef = np.ravel(theta[:-1, 0])  # for all but last column,  only first col
    else:
        intercept = 0.0
        coef = np.ravel(theta[:, 0])  # all rows, only first col

    return coef, intercept, rss_clean, h_subset

# currently support for np.ndarray and matrix

def validate(X, y, use_intercept):
    if X.ndim == 1:
        X = np.reshape(X, [X.shape[0], 1])
    if y.ndim == 1:
        y = np.reshape(y, [y.shape[0], 1])

    if type(X) is not np.ndarray:
        X = np.ndarray(X)
    if type(y) is not np.ndarray:
        y = np.ndarray(y)

    if use_intercept:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    return X, y



def fast_lts_to_feasible_solution():
    return


def fast_lts_to_feasible_to_bab():
    return

