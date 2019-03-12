import lts.fastlts.fast_lts as fastlts
from data.data_generator import generate_data_ND
from scipy import spatial


def test_cpp(n=100, p=2):
    print('test started...')
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = fastlts.FLTSRegressorCPP()
    lts.fit(x, y, use_intercept=True, num_starts=10)

    # lts
    weights_lts = lts.coef_

    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # OLS on the clean data
    lts.fit(x_clean, y_clean, use_intercept=True, h_size=x_clean.shape[0])
    weights_global_minimum = lts.coef_

    # cos similarity
    result = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)
    print('cos: ', result)
    print('...test finished')


def test_numpy(n=100, p=2):
    print('test started...')
    x, y, x_clean, y_clean = generate_data_ND(n, p)
    lts = fastlts.FLTSRegressor()
    lts.fit(x, y, use_intercept=True, num_starts=10)

    # lts
    weights_lts = lts.coef_

    # print data
    print('rss: ', lts.rss_)
    print('itr: ', lts.n_iter_)
    print('sec: ', lts.time_total_)

    # OLS on the clean data
    lts.fit(x_clean, y_clean, use_intercept=True, h_size=x_clean.shape[0])
    weights_global_minimum = lts.coef_

    # cos similarity
    result = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)
    print('cos: ', result)
    print('...test finished')
