# import sys
# print(sys.path)

from tests.feasible import test_dev as feasible_test
from tests.fastlts import test_dev as fast_lts_test


def run_tests_feasible_solution():
    feasible_test.test_cpp()


def run_tests_fast_lts():
    fast_lts_test.test_cpp(n=10000, p=5)


if __name__ == '__main__':
    # run_tests_feasible_solution()
    run_tests_fast_lts()
