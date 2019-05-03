import lts.feasible.feasible_solution as feasible
from data.data_generator import generate_data_ND
from scipy import spatial


def test_cpp(n=100, p=2, algorithm='fsa', calculation='inv', num_starts=10, max_steps=50, intercept=True, h_size='default'):
    print('test started...')
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = feasible.FSRegressorCPP(num_starts=num_starts, max_steps=max_steps, use_intercept=intercept,
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
    lts = feasible.FSRegressor(num_starts=10, max_steps=50, use_intercept=True,
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
