# import importlib
# moduleName = input('Enter module name:')
# importlib.import_module(moduleName)

import sys
print(sys.path)

import src.feasible_solution as fs_lts
# works with pycharm from .src import feasible_solution_cpp as fs_lts_cpp
import src.feasible_solution_cpp as fs_lts_cpp  # works with cmd

from data_generator import generate_data_ND
from scipy import spatial



if __name__ == '__main__':
    n = 1000
    p = 3
    print('start')
    X, y, X_clean, y_clean = generate_data_ND(n, p)
    lts = fs_lts_cpp.FeasibleSolutionRegressionCPP()
    lts.fit(X, y, use_intercept=True, num_starts=10)

    # lts
    weights_correct = lts.coef_

    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('tim: ', lts.time_total_)

    # OLS on the clean data
    lts.fit(X_clean, y_clean, use_intercept=True, h_size=X_clean.shape[0])
    weights_expected = lts.coef_
    # print('rsO: ', lts.rss_)
    # cos similarity
    result = 1 - spatial.distance.cosine(weights_correct, weights_expected)
    print('cos: ', result)

#  module -  file.py
#  built-in module - c code compiled into Python interpret, no .py
#  package - folder with __init__.py (from 3.3 even without)

#  imported module  - python run code in module file
#  imported package - python run code in __init__.py
#  build in modules can be overwritten
    # import math .. imports pythons math, not math.py form same dir
    # import random .. if random.py in current dir, it is preferred
#  import is case sensitive
#  sys.path
    #  list of paths (strings) that specifies search path for modules
    # initialized from env. variable PYTHONPATH + install-dependent default
    # sys.path[0] .. directory containing script used to invoke Python interpret
    # if not available (script from stdin || interpret invoked interactively)
    # then sys.path[0] is empty string -> python search current dir

    # 1. IMPORTANT
    # sys.path doesn't care about current work directory
    #   /home/folder/test$ python ./packA/subA/subA1.py
    #       sys.path[0] := /home/folder/test/packA/subA

    # 2. IMPORTANT
    # sys.path is shared across all imported modules
    # test/start.py -> imports mod.py
    # test/other.py
    # test/packageA/mod.py -> it prints sys.path
    # python ./start.py
    #   ../../test/
    #   THAT MEANS: mod.py can import other.py because it is in path !!!



#





















