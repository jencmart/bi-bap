import lts.feasible.feasible_solution as feasible
from data.data_generator import generate_data_ND
import lts.fastlts.fast_lts as fastlts
from scipy import spatial
import numpy as np

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
    print('test started...')
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


def fast_lts_to_feasible_solution():
    return


def fast_lts_to_feasible_to_bab():
    return

