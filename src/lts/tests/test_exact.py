import lts.exact.exact as exact
import lts.feasible.feasible_solution as feasible
from data.data_generator import generate_data_ND
from scipy import spatial
import data.data_generator as generator
import numpy as np


def test_numpy(n=100, p=2, algorithm='bab', use_intercept=True, index_subset=None):

    print('test started...')

    # fit exact exhaustive algorithm
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h1 = lts.h_subset_
    h1.sort()
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)


def test_BAB_vs_RAND_BSA_cpp(n=20, p=2, use_intercept=True):
    print('test started...({} x {})'.format(n, p))

    # GENERATE THE DATA
    x, y, x_clean, y_clean = generator.generate_dataset_simple(n, p, outlier_ratio=0.3)

    # BSA
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='bab', calculation='inv')
    lts.fit(x, y)
    h1 = lts.h_subset_
    print('BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # RANDOM BSA

    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='rbsa', calculation='inv',
                                     num_starts=500)
    lts.fit(x, y)
    h2 = lts.h_subset_
    print('RANDOM BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    print('------')
    if not np.array_equal(h1, h2):
        print(' not equal')


def test_feasible_vs_RAND_BSA_cpp(n=20, p=2, use_intercept=True):
    print('test started...({} x {})'.format(n, p))

    # GENERATE THE DATA
    x, y, x_clean, y_clean = generator.generate_dataset_simple(n, p, outlier_ratio=0.3)

    # BSA
    lts = feasible.LTSRegressorFeasibleCPP(use_intercept=use_intercept, algorithm='fsa', calculation='inv')
    lts.fit(x, y)
    h1 = lts.h_subset_
    print('feasible moea qr:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # RANDOM BSA

    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='rbsa', calculation='inv',
                                     num_starts=500)
    lts.fit(x, y)
    h2 = lts.h_subset_
    print('RANDOM BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)


def test_all(n=20, p=2, use_intercept=True, use_subset=False):

    print('test started...({} x {})'.format(n, p))

    # GENERATE THE DATA
    x, y, x_clean, y_clean = generator.generate_dataset_simple(n, p, outlier_ratio=0.3)

    # Exhaustive
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='exa', calculation='inv')
    lts.fit(x, y)
    h1 = lts.h_subset_
    print('Exhaustive:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # if we are testing subset, set it here
    index_subset = None
    if use_subset:
        index_subset = h1

    # Bab
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='bab', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h2 = lts.h_subset_
    print('Bab:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # BSA
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='bsa', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h3 = lts.h_subset_
    print('BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # print subsets
    print(h1)
    print(h2)
    print(h3)

    if not np.array_equal(h1, h2) or not np.array_equal(h2, h3):
        print(' YOU ARE FUCKED numpy fucked')
        exit(99)

    print('numpy finished... staring CPP')

    # Exhaustive
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='exa', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h1 = lts.h_subset_
    print('Exhaustive:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # Bab
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='bab', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h2 = lts.h_subset_
    print('Bab:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # BSA
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='bsa', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h3 = lts.h_subset_
    print('BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # print subsets
    print(h1)
    print(h2)
    print(h3)

    if not np.array_equal(h1, h2) or not np.array_equal(h2, h3):
        print(' YOU ARE FUCKED cpp fucked')
        exit(99)

    print('...testing random BSA')

    # RANDOM BSA PYT
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='rbsa', calculation='inv',
                                  num_starts=500)
    lts.fit(x, y)
    h4 = lts.h_subset_
    print('rand_BSA:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # RANDOM BSA CPP
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='rbsa', calculation='inv',
                                     num_starts=500)
    lts.fit(x, y)
    h5 = lts.h_subset_
    print('rand_BSA_CPP:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    print(h4)
    print(h5)

    print('...test finished')


def test_numpy_combined(n=100, p=2, algorithm='bab', use_intercept=True, index_subset=None):
    print('test started...({} x {})'.format(n, p))

    # fit exact exhaustive algorithm
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm='exa', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h1 = lts.h_subset_
    h1.sort()
    print('Exhaustive:')
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # fit exact defined algorithm
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h2 = lts.h_subset_
    h2.sort()
    print('custom ('+algorithm+'):')
    print('cuts: ', lts.n_iter_)
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    # fit exact bab algorithm
    lts = exact.LTSRegressorExact(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
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


def test_cpp(n=100, p=2, algorithm='bab', use_intercept=True, index_subset=None):
    print('test started...')

    # fit exact exhaustive algorithm
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm=algorithm, calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h1 = lts.h_subset_
    h1.sort()
    h1 = np.asarray(h1)
    print('CPP ('+algorithm+"):")
    print('rss: ', lts.rss_)
    print('cuts: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # fit exact bab algorithm [NUMPY - TESTING]
    lts = exact.LTSRegressorExactCPP(use_intercept=use_intercept, algorithm='bab', calculation='inv')
    lts.fit(x, y, index_subset=index_subset)
    h3 = lts.h_subset_
    h3.sort()
    h3 = np.asarray(h3)
    print('cpp BAB:')
    print('cuts: ', lts.n_iter_)
    print('rss: ', lts.rss_)
    print('sec: ', lts.time_total_)

    print('-------------')
    print(h1)
    print(h3)
