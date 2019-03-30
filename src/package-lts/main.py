# import sys
# print(sys.path)

from tests.feasible import test_dev as feasible_test
from tests.feasibleoe import test_dev as feasible_oe_test

from tests.fastlts import test_dev as fast_lts_test
from experiments import speed_fast_lts as lts_speed


def test_feasible_solution():
    feasible_test.test_cpp()


def test_feasible_OE_solution():
    feasible_oe_test.test_numpy(n=30, p=3)

def test_fast_lts():
    fast_lts_test.test_cpp(n=10000, p=5)


def experiment_fast_lts_speed():
    lts_speed.fast_lts_cpp_vs_numpy()


if __name__ == '__main__':
    # run_tests_feasible_solution()
    # test_fast_lts()
    # experiment_fast_lts_speed()
    # lts_speed.fast_lts_cpp_only()
    test_feasible_OE_solution()
    #lts_speed.fast_lts_cpp_big()

