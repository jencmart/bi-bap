# import sys
# print(sys.path)

from tests import test_feasible as feasible_test
from tests import test_exact as exact_test

from tests import test_fast as fast_lts_test
from experiments import speed_fast_lts as lts_speed
from experiments import experiments_probabilistic

# Feasible solution
def test_feasible_solution_numpy():
    feasible_test.test_numpy(n=10, p=2, algorithm='mmea', calculation='qr')
    # mmea-qr 1000 x 3  sec:  272.21239226800003


def test_feasible_solution_numpy_starting_subset():
    feasible_test.test_fast_feasible_numpy(n=20, p=2, algorithm='mmea', calculation='inv')
    # mmea-qr 1000 x 3  sec:  272.21239226800003


# Feasible solution
def test_feasible_solution_cpp():
    feasible_test.test_cpp(n=200, p=1, algorithm='fsa', calculation='qr', num_starts=1, max_steps=500)
    # mmea-qr 10 000 x 3 sec:  357.7933044433594


# Exact algorithms
def test_exact_numpy():
    # exact_test.test_numpy(n=10, p=2, algorithm='bab')
    # exact_test.test_all(n=15, p=3, use_intercept=True, use_subset=True)
    # exact_test.test_BAB_vs_RAND_BSA_cpp(n=50, p=2, use_intercept=True)
    exact_test.test_feasible_vs_RAND_BSA_cpp(n=1000, p=2, use_intercept=True)

# Feasible solution OE QR
def test_exact_cpp():

    exact_test.test_cpp(n=10, p=2, use_intercept=True, algorithm='exa')
    # p=2(+1)      32         33        40      50      55      60
    # cpp bab      2.5s       2.1       4.8     117     394

    # bsa 4s   (100x2)
    # bsa 70s  (200x2)
    # bsa 358s (300x2)


# Fast LTS
def test_fast_lts_cpp():
    fast_lts_test.test_cpp(n=10, p=2)


def test_fast_lts_numpy():
    fast_lts_test.test_numpy(n=10, p=2)


def experiment_fast_lts_speed():
    lts_speed.fast_lts_cpp_vs_numpy()


if __name__ == '__main__':
    # run_tests_feasible_solution()
    # test_fast_lts()
    # experiment_fast_lts_speed()
    # lts_speed.fast_lts_cpp_only()
    # test_feasible_OE_solution_numpy()
    # test_exact_numpy()
    # test_feasible_solution_numpy()
    # test_exact_cpp()
    # experiments_probabilistic.experiment_speed_probabilistic('./out/experiment_probabilistic_p3.csv')
    # test_feasible_solution_numpy_starting_subset()
    # test_feasible_solution_cpp()
    # test_exact_numpy()
    # experiments_probabilistic.experiment_speed_probabilistic_big('./out/experiment_probabilistic_big_p2.csv')
    # experiments_probabilistic.experiment_lts(output='./out/lts_cnt_convergence.csv')

    #feasible_test.test_fast_and_feasible(n=28, p=3)
    # test_feasible_solution_numpy()
    lts_speed.fast_lts_cpp_big()

