from scipy import spatial
import cppimport.import_hook
from lts.fastlts import fast_lts
from data import data_generator as data_gen
import pandas as pd


def get_regressor(typ):
    if typ == 'ltsPython':
        return fast_lts.FLTSRegressor()
    if typ == 'ltsCpp':
        return fast_lts.FLTSRegressorCPP()


# it would be best to pass experiment class
# to generic experiment processor
def fast_lts_cpp_vs_numpy(output='./out/fast_lts_cpp_vs_numpy_results.csv'):
    types = ['ltsPython', 'ltsCpp']
    inter = True

    # same as in FAST-LTS paper -- so its a benchmark, not experiment
    experiments = [(100, 2), (100, 3), (100, 5), (500, 2), (500, 3), (500, 5), (1000, 2),
                   (1000, 5), (1000, 10), (10000, 2), (10000, 5), (10000, 10), (50000, 2), (50000, 5)]

    cnt_experiments = len(experiments)

    res = pd.DataFrame(columns=['implementation', 'n', 'p', 'rss', 'iter', 'time', 'cos_ols', 'intercept'])
    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt, experiment in enumerate(experiments):
        n, p = experiment
        x, y, x_clean, y_clean = data_gen.generate_data_ND(n, p)
        for t in types:
            lts = get_regressor(t)
            print('running...[{}/{}][{}]'.format(cnt + 1, cnt_experiments, t))

            # lts
            lts.fit(x, y, use_intercept=True)
            weights_lts = lts.coef_
            iters = lts.n_iter_
            time = lts.time_total_
            rss = lts.rss_

            # ols
            lts.fit(x_clean, y_clean, use_intercept=inter, h_size=x_clean.shape[0])
            weights_global_minimum = lts.coef_
            # cos similarity
            cos = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)

            res = res.append(pd.Series([t, n, p, rss, iters, time, cos, inter], index=res.columns), ignore_index=True)

        # save experiments to file [prevent loosing data on failure]
        res.to_csv(output)


def fast_lts_cpp_only(output='./out/fast_lts_cpp_results.csv'):
    types = ['ltsCpp']
    inter = True

    # same as in FAST-LTS paper -- so its a benchmark, not experiment
    experiments = [(100, 2), (100, 3), (100, 5), (500, 2), (500, 3), (500, 5), (1000, 2),
                   (1000, 5), (1000, 10), (10000, 2), (10000, 5), (10000, 10), (50000, 2), (50000, 5), (50000, 10),
                   (100000, 5), (100000, 10), (500000, 5), (500000, 10), (1000000, 5), (1000000, 10),
                   (2500000, 5), (2500000, 10), (3000000, 2), (3000000, 3), (3000000, 5)]

    cnt_experiments = len(experiments)

    res = pd.DataFrame(columns=['implementation', 'n', 'p', 'rss', 'iter', 'time', 'cos_ols', 'intercept'])
    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt, experiment in enumerate(experiments):
        n, p = experiment
        x, y, x_clean, y_clean = data_gen.generate_data_ND(n, p)
        for t in types:
            lts = get_regressor(t)
            print('running...[{}/{}][{}]'.format(cnt + 1, cnt_experiments, t))

            # lts
            lts.fit(x, y, use_intercept=True)
            weights_lts = lts.coef_
            iters = lts.n_iter_
            time = lts.time_total_
            rss = lts.rss_

            # ols
            lts.fit(x_clean, y_clean, use_intercept=inter, h_size=x_clean.shape[0])
            weights_global_minimum = lts.coef_
            # cos similarity
            cos = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)

            res = res.append(pd.Series([t, n, p, rss, iters, time, cos, inter], index=res.columns), ignore_index=True)

        # save experiments to file [prevent loosing data on failure]
        res.to_csv(output)


def fast_lts_cpp_big(output='./out/fast_lts_cpp_big_results.csv'):
    types = ['ltsCpp']
    inter = True

    # same as in FAST-LTS paper -- so its a benchmark, not experiment
    experiments = [(3500000, 5), (4000000, 2), (4000000, 5)]

    cnt_experiments = len(experiments)

    res = pd.DataFrame(columns=['implementation', 'n', 'p', 'rss', 'iter', 'time', 'cos_ols', 'intercept'])
    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt, experiment in enumerate(experiments):
        n, p = experiment
        x, y, x_clean, y_clean = data_gen.generate_data_ND(n, p)
        for t in types:
            lts = get_regressor(t)
            print('running...[{}/{}][{}]'.format(cnt + 1, cnt_experiments, t))

            # lts
            lts.fit(x, y, use_intercept=True)
            weights_lts = lts.coef_
            iters = lts.n_iter_
            time = lts.time_total_
            rss = lts.rss_

            # ols
            lts.fit(x_clean, y_clean, use_intercept=inter, h_size=x_clean.shape[0])
            weights_global_minimum = lts.coef_
            # cos similarity
            cos = 1 - spatial.distance.cosine(weights_lts, weights_global_minimum)

            res = res.append(pd.Series([t, n, p, rss, iters, time, cos, inter], index=res.columns), ignore_index=True)

        # save experiments to file [prevent loosing data on failure]
        res.to_csv(output)
