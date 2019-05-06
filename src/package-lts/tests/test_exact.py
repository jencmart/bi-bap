import lts.exact.exact as exact
from data.data_generator import generate_data_ND
from scipy import spatial
import numpy as np


def test_numpy(n=100, p=2, algorithm='bab', use_intercept=True, rss=None):

    print('test started...')

    # fit exact exhaustive algorithm
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='exa', calculation='inv')
    lts.fit(x, y, set_rss=rss)
    h1 = lts.h_subset_
    h1.sort()
    print('Exhaustive:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # fit exact defined algorithm
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, set_rss=rss)
    h2 = lts.h_subset_
    h2.sort()
    print('custom ('+algorithm+'):')
    print('cuts: ', lts.n_iter_)
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # fit exact bab algorithm
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, set_rss=rss)
    h3 = lts.h_subset_
    h3.sort()
    print('BAB:')
    print('cuts: ', lts.n_iter_)
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # print sorted subsets
    print(h1)
    print(h2)
    print(h3)
    if not np.array_equal(h1, h2) or not np.array_equal(h2, h3):
        print(' YOU ARE FUCKED')
        exit(99)

    print('...test finished')


def test_cpp(n=100, p=2, algorithm='bab', use_intercept=True, rss=None):
    print('test started...')

    # fit exact exhaustive algorithm
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='bsa', calculation='inv')
    lts.fit(x, y, set_rss=rss)
    h1 = lts.h_subset_
    h1.sort()
    h1 = np.asarray(h1)
    print('CPP ('+algorithm+"):")
    print('rss: ', lts.rss_)
    print('cuts: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # fit exact bab algorithm [NUMPY - TESTING]
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, set_rss=rss)
    h3 = lts.h_subset_
    h3.sort()
    print('numpy BAB:')
    print('cuts: ', lts.n_iter_)
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    print('-------------')
    print(h1)
    print(h3)
