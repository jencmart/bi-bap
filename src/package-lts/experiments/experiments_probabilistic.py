import lts.feasible.feasible_solution as feasible
import data.data_generator as generator
import lts.fastlts.fast_lts as fastlts
import lts.exact.exact as exact
import pandas as pd
from scipy import spatial
import numpy as np
from scipy import linalg


# 'FAST-LTS', 'FSA-I', 'FSA-QR', 'MOEA-I', 'MOEA-QR', 'MMEA-I', 'MMEA-QR'
def get_algorithm(alg, max_steps, intercept):
    if alg == 'FAST-LTS':
        return fastlts.LTSRegressorFastCPP(num_starts=500, num_initial_c_steps=2, num_starts_to_finish=10,
                                           max_steps=50, threshold=1e-6,
                                           use_intercept=intercept)
    if alg == 'FSA-I':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='fsa', calculation='inv')
    if alg == 'FSA-QR':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='fsa', calculation='qr')
    if alg == 'MOEA-I':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='moea', calculation='inv')
    if alg == 'MOEA-QR':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='moea', calculation='qr')
    if alg == 'MMEA-I':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='mmea', calculation='inv')
    if alg == 'MMEA-QR':
        return feasible.LTSRegressorFeasibleCPP(num_starts=1, max_steps=max_steps, use_intercept=intercept,
                                                algorithm='mmea', calculation='qr')


def experiment_speed_probabilistic(output='./out/experiment_probabilistic_p1.csv'):
    experiments = [(20, 2),
                   (100, 3),
                   (100, 5),
                   (500, 2),
                   (500, 5)
                   ]

    data_sets = [(0.1, 0.0), (0.3, 0.0), (0.45, 0.0),
                 (0.1, 1), (0.3, 1), (0.45, 1),
                 (0.1, 0.4), (0.3, 0.4), (0.45, 0.4)]

    algorithms = ['FAST-LTS', 'FSA-I', 'FSA-QR', 'MOEA-I', 'MOEA-QR', 'MMEA-I', 'MMEA-QR']

    max_steps = 100
    intercept = True
    cnt_experiments = len(experiments)
    cnt_data_sets = len(data_sets)
    cnt_algorithms = len(algorithms)
    num_starts = 100
    h_size = 'default'
    leverage_ratio = 0.2

    res = pd.DataFrame(columns=['algorithm', 'n', 'p', 'out', 'out_2model',
                                'rss', 'iter', 'time', 'cos',
                                'intercept_diff', 'l2', 'global_min', 'intercept', 'h_size', 'max_steps',
                                'leverage_ratio'])

    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt_dataset, dataset in enumerate(data_sets):  # for all data sets

        for cnt_experiment, experiment in enumerate(experiments):  # run all experiments

            for i in range(num_starts):  # generate the data 100 times

                # MODEL
                # model X ~ N(m,s)
                x_m = 0
                x_s = 10

                # errors e~N(m,s)
                e_m = 0
                e_s = np.random.randint(low=1, high=10)

                # errors outliers e~N(m,s) or e~Exp(s)
                e_out_m = np.random.randint(low=-50, high=50)
                e_out_s = np.random.randint(low=50, high=200)

                # leverage points
                x_lav_m = np.random.randint(low=10, high=50)
                x_lav_s = np.random.randint(low=10, high=50)

                # SECOND MODEL
                # second model X ~ N(m,s)
                x2_m = np.random.randint(low=-10, high=10)
                x2_s = 10

                # second model errors e~N(m,s)
                e2_m = 0
                e2_s = np.random.randint(low=5, high=10)

                if np.random.rand() >= 0.5:
                    e_out_dist = 'n'
                else:
                    e_out_dist = 'e'

                outlier_ratio, outlier_second_model_ratio = dataset

                n, p = experiment

                X, y, X_clean, y_clean = generator.generate_dataset(n, p,  # n x p
                                                                    outlier_ratio=outlier_ratio,
                                                                    # ratio of the outliers in the whole data set
                                                                    leverage_ratio=leverage_ratio,
                                                                    # ratio of data outlying in x , in the whole dataset
                                                                    x_ms=(x_m, x_s),  # not outlying x  ~ N(mean, sd)
                                                                    x_lav_ms=(x_lav_m, x_lav_s),
                                                                    # outlying x  ~ N(mean, sd)
                                                                    e_ms=(e_m, e_s),  # not outlying y  e ~ N(mean, sd)
                                                                    e_out_ms=(e_out_m, e_out_s),
                                                                    # outlying y   e ~ N(mean, std)  or ~Exp(std) 'n''e'
                                                                    e_out_dist=e_out_dist,
                                                                    # n/ln/e distribution of e for outlying y
                                                                    outlier_secon_model_ratio=outlier_second_model_ratio,
                                                                    # ratio of outliers which are not outlying in (y)
                                                                    # but instead are form completely different model
                                                                    # (if 0, data only from one model)
                                                                    # coeff_scale=coef_scale,
                                                                    # random vector of regression coefficients
                                                                    # c \in { (-coeff_scale, coeff_cale)^p } \ 0  ,
                                                                    # so that yi = c xi.T + e
                                                                    mod2_x_ms=(x2_m, x2_s),
                                                                    mod2_e_ms=(e2_m, e2_s))

                for cnt_alg, alg in enumerate(algorithms):  # on each algorithm
                    print('running...dataset[{}/{}] experiment[{}/{}]({}x{}) algorithm[{}/{}] for[{}/{}] '.format(
                        cnt_dataset + 1, cnt_data_sets, cnt_experiment + 1, cnt_experiments, n, p, cnt_alg + 1,
                        cnt_algorithms,
                        i + 1, num_starts), end='')

                    # Construct the algorithm
                    lts = get_algorithm(alg, max_steps=max_steps, intercept=intercept)

                    # FIT LTS
                    lts.fit(X, y, h_size=h_size)
                    weights_lts = lts.coef_
                    intercept_lts = lts.intercept_
                    iters = lts.n_iter_
                    time = lts.time_total_
                    rss = lts.rss_
                    subset = lts.h_subset_
                    subset.sort()
                    subset = np.asarray(subset)

                    lts_h_size = subset.shape[0]

                    print('t: {0:.2f}'.format(time))

                    # OLS on the clean data
                    weights_clean, intercept_clean, rss_clean, subset_clean = solve_clean(X_clean, y_clean, lts_h_size,
                                                                                          intercept)
                    # Calculate some similarity

                    # cos similarity
                    cos_sim = 1 - spatial.distance.cosine(weights_lts, weights_clean)
                    intercept_diff = abs(intercept_lts - intercept_clean)

                    # l2 distance
                    if intercept:
                        l2_dist = linalg.norm(np.append(weights_lts, [intercept_lts]) - np.append(weights_clean,
                                                                                                  [intercept_clean]))
                    else:
                        l2_dist = linalg.norm(weights_lts - weights_clean)

                    if np.array_equal(subset, subset_clean):
                        global_min = True
                    else:
                        global_min = False

                    res = res.append(pd.Series([alg, n, p, outlier_ratio, outlier_second_model_ratio, rss, iters, time,
                                                cos_sim, intercept_diff, l2_dist, global_min, intercept, lts_h_size,
                                                max_steps, leverage_ratio], index=res.columns), ignore_index=True)

                    # save experiments to file [prevent loosing data on failure]
                    res.to_csv(output)


def experiment_speed_probabilistic_big(output='./out/experiment_probabilistic_big.csv'):
    experiments = [
        (1000, 2),
        (1000, 5),
        (1000, 10),
        (1000, 20)
    ]

    data_sets = [(0.1, 0.0), (0.3, 0.0), (0.45, 0.0),
                 (0.1, 1), (0.3, 1), (0.45, 1),
                 (0.1, 0.4), (0.3, 0.4), (0.45, 0.4)]

    algorithms = ['FAST-LTS', 'MOEA-I', 'MOEA-QR', 'MMEA-I', 'MMEA-QR']

    max_steps = 100
    intercept = True
    cnt_experiments = len(experiments)
    cnt_data_sets = len(data_sets)
    cnt_algorithms = len(algorithms)
    num_starts = 100
    h_size = 'default'
    leverage_ratio = 0.2

    res = pd.DataFrame(columns=['algorithm', 'n', 'p', 'out', 'out_2model',
                                'rss', 'iter', 'time', 'cos',
                                'intercept_diff', 'l2', 'global_min', 'intercept', 'h_size', 'max_steps',
                                'leverage_ratio'])

    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt_dataset, dataset in enumerate(data_sets):  # for all data sets

        for cnt_experiment, experiment in enumerate(experiments):  # run all experiments

            for i in range(num_starts):  # generate the data 100 times

                # MODEL
                # model X ~ N(m,s)
                x_m = 0
                x_s = 10

                # errors e~N(m,s)
                e_m = 0
                e_s = np.random.randint(low=1, high=10)

                # errors outliers e~N(m,s) or e~Exp(s)
                e_out_m = np.random.randint(low=-50, high=50)
                e_out_s = np.random.randint(low=50, high=200)

                # leverage points
                x_lav_m = np.random.randint(low=10, high=50)
                x_lav_s = np.random.randint(low=10, high=50)

                # SECOND MODEL
                # second model X ~ N(m,s)
                x2_m = np.random.randint(low=-10, high=10)
                x2_s = 10

                # second model errors e~N(m,s)
                e2_m = 0
                e2_s = np.random.randint(low=5, high=10)

                if np.random.rand() >= 0.5:
                    e_out_dist = 'n'
                else:
                    e_out_dist = 'e'

                outlier_ratio, outlier_second_model_ratio = dataset

                n, p = experiment

                X, y, X_clean, y_clean = generator.generate_dataset(n, p,  # n x p
                                                                    outlier_ratio=outlier_ratio,
                                                                    # ratio of the outliers in the whole data set
                                                                    leverage_ratio=leverage_ratio,
                                                                    # ratio of data outlying in x , in the whole dataset
                                                                    x_ms=(x_m, x_s),  # not outlying x  ~ N(mean, sd)
                                                                    x_lav_ms=(x_lav_m, x_lav_s),
                                                                    # outlying x  ~ N(mean, sd)
                                                                    e_ms=(e_m, e_s),  # not outlying y  e ~ N(mean, sd)
                                                                    e_out_ms=(e_out_m, e_out_s),
                                                                    # outlying y   e ~ N(mean, std)  or ~Exp(std) 'n''e'
                                                                    e_out_dist=e_out_dist,
                                                                    # n/ln/e distribution of e for outying y
                                                                    outlier_secon_model_ratio=
                                                                    outlier_second_model_ratio,
                                                                    # ratio of outlers which are not outling in (y)
                                                                    # but instead are form comletely different model
                                                                    # (if 0, data only from one model)
                                                                    # coeff_scale=coef_scale,
                                                                    # random vector of regression coefficients
                                                                    # c \in { (-coeff_scale, coeff_cale)^p } \ 0  ,
                                                                    # so that yi = c xi.T + e
                                                                    mod2_x_ms=(x2_m, x2_s),
                                                                    mod2_e_ms=(e2_m, e2_s))

                for cnt_alg, alg in enumerate(algorithms):  # on each algorithm
                    print('running...dataset[{}/{}] experiment[{}/{}]({}x{}) algorithm[{}/{}] for[{}/{}] '.format(
                        cnt_dataset + 1, cnt_data_sets, cnt_experiment + 1, cnt_experiments, n, p, cnt_alg + 1,
                        cnt_algorithms,
                        i + 1, num_starts), end='')

                    # Construct the algorithm
                    lts = get_algorithm(alg, max_steps=max_steps, intercept=intercept)

                    # FIT LTS
                    lts.fit(X, y, h_size=h_size)
                    weights_lts = lts.coef_
                    intercept_lts = lts.intercept_
                    iters = lts.n_iter_
                    time = lts.time_total_
                    rss = lts.rss_
                    subset = lts.h_subset_
                    subset.sort()
                    subset = np.asarray(subset)

                    lts_h_size = subset.shape[0]

                    print('t: {0:.2f}'.format(time))

                    # OLS on the clean data
                    weights_clean, intercept_clean, rss_clean, subset_clean = solve_clean(X_clean, y_clean, lts_h_size,
                                                                                          intercept)
                    # Calculate some similarity

                    # cos similarity
                    cos_sim = 1 - spatial.distance.cosine(weights_lts, weights_clean)
                    intercept_diff = abs(intercept_lts - intercept_clean)

                    # l2 distance
                    if intercept:
                        l2_dist = linalg.norm(np.append(weights_lts, [intercept_lts]) - np.append(weights_clean,
                                                                                                  [intercept_clean]))
                    else:
                        l2_dist = linalg.norm(weights_lts - weights_clean)

                    if np.array_equal(subset, subset_clean):
                        global_min = True
                    else:
                        global_min = False

                    res = res.append(pd.Series([alg, n, p, outlier_ratio, outlier_second_model_ratio, rss, iters, time,
                                                cos_sim, intercept_diff, l2_dist, global_min, intercept, lts_h_size,
                                                max_steps, leverage_ratio], index=res.columns), ignore_index=True)

                    # save experiments to file [prevent loosing data on failure]
                    res.to_csv(output)


# 'BAB', 'MOEA-QR-BAB', 'MOEA-QR-BSA', 'BSA', 'EXH']

def fit_algorithm_exact(alg, intercept, X, y, h_size, num_starts=100, max_steps=500):
    if alg == 'BAB':
        # BAB
        lts = exact.LTSRegressorExactCPP(use_intercept=intercept, algorithm='bab', calculation='inv')
        lts.fit(X, y, h_size=h_size)
        return lts

    if alg == 'MOEA-QR-BAB':
        # MOEA-QR
        lts = feasible.LTSRegressorFeasibleCPP(num_starts=num_starts, max_steps=max_steps, use_intercept=intercept,
                                               algorithm='moea', calculation='qr')
        lts.fit(X, y, h_size=h_size)
        index_subset = lts.h_subset_

        # BAB
        lts = exact.LTSRegressorExactCPP(use_intercept=intercept, algorithm='bab', calculation='inv')
        lts.fit(X, y, index_subset=index_subset)
        return lts

    if alg == 'MOEA-QR-BSA':
        # MOEA-QR
        lts = feasible.LTSRegressorFeasibleCPP(num_starts=num_starts, max_steps=max_steps, use_intercept=intercept,
                                               algorithm='moea', calculation='qr')
        lts.fit(X, y, h_size=h_size)
        index_subset = lts.h_subset_

        # BSA
        lts = exact.LTSRegressorExactCPP(use_intercept=intercept, algorithm='bsa', calculation='inv')
        lts.fit(X, y, index_subset=index_subset, h_size=h_size)
        return lts

    if alg == 'BSA':
        # BSA
        lts = exact.LTSRegressorExactCPP(use_intercept=intercept, algorithm='bsa', calculation='inv')
        lts.fit(X, y, h_size=h_size)
        return lts

    if alg == 'EXACT':
        # EXH
        lts = exact.LTSRegressorExactCPP(use_intercept=intercept, algorithm='exa', calculation='inv')
        lts.fit(X, y, h_size=h_size)
        return lts


def experiment_speed_exact(output='./out/experiment_exact.csv'):
    experiments = [
        (15, 4),
        (20, 3),
        (20, 4),
        (25, 3),
        (30, 2),
        (30, 3)
    ]

    data_sets = [(0.1, 0.0), (0.3, 0.0), (0.45, 0.0),
                 (0.1, 1), (0.3, 1), (0.45, 1),
                 (0.1, 0.4), (0.3, 0.4), (0.45, 0.4)]

    algorithms = ['BAB', 'MOEA-QR-BAB', 'MOEA-QR-BSA', 'BSA', 'EXACT']

    probabilistic_starts = 500
    probabilistic_steps = 500

    max_steps = 100
    intercept = True
    cnt_experiments = len(experiments)
    cnt_data_sets = len(data_sets)
    cnt_algorithms = len(algorithms)
    num_starts = 100
    h_size = 'default'
    leverage_ratio = 0.2

    res = pd.DataFrame(columns=['algorithm', 'n', 'p', 'out', 'out_2model',
                                'rss', 'iter', 'time', 'intercept', 'h_size', 'max_steps',
                                'leverage_ratio'])

    print('starting experiments [{}] ...'.format(cnt_experiments))

    for cnt_experiment, experiment in enumerate(experiments):  # run all experiments

        for cnt_dataset, dataset in enumerate(data_sets):  # for all data sets

            for i in range(num_starts):  # generate the data 100 times

                # MODEL
                # model X ~ N(m,s)
                x_m = 0
                x_s = 10

                # errors e~N(m,s)
                e_m = 0
                e_s = np.random.randint(low=1, high=10)

                # errors outliers e~N(m,s) or e~Exp(s)
                e_out_m = np.random.randint(low=-50, high=50)
                e_out_s = np.random.randint(low=50, high=200)

                # leverage points
                x_lav_m = np.random.randint(low=10, high=50)
                x_lav_s = np.random.randint(low=10, high=50)

                # SECOND MODEL
                # second model X ~ N(m,s)
                x2_m = np.random.randint(low=-10, high=10)
                x2_s = 10

                # second model errors e~N(m,s)
                e2_m = 0
                e2_s = np.random.randint(low=5, high=10)

                if np.random.rand() >= 0.5:
                    e_out_dist = 'n'
                else:
                    e_out_dist = 'e'

                outlier_ratio, outlier_second_model_ratio = dataset

                n, p = experiment

                X, y, X_clean, y_clean = generator.generate_dataset(n, p,  # n x p
                                                                    outlier_ratio=outlier_ratio,
                                                                    # ratio of the outliers in the whole data set
                                                                    leverage_ratio=leverage_ratio,
                                                                    # ratio of data outlying in x , in the whole dataset
                                                                    x_ms=(x_m, x_s),  # not outlying x  ~ N(mean, sd)
                                                                    x_lav_ms=(x_lav_m, x_lav_s),
                                                                    # outlying x  ~ N(mean, sd)
                                                                    e_ms=(e_m, e_s),  # not outlying y  e ~ N(mean, sd)
                                                                    e_out_ms=(e_out_m, e_out_s),
                                                                    # outlying y   e ~ N(mean, std)  or ~Exp(std) 'n''e'
                                                                    e_out_dist=e_out_dist,
                                                                    # n/ln/e distribution of e for outlying y
                                                                    outlier_secon_model_ratio=outlier_second_model_ratio,
                                                                    # ratio of outliers which are not outlying in (y)
                                                                    # but instead are form completely different model
                                                                    # (if 0, data only from one model)
                                                                    # coeff_scale=coef_scale,
                                                                    # random vector of regression coefficients
                                                                    # c \in { (-coeff_scale, coeff_cale)^p } \ 0,
                                                                    # so that yi = c xi.T + e
                                                                    mod2_x_ms=(x2_m, x2_s),
                                                                    mod2_e_ms=(e2_m, e2_s))

                for cnt_alg, alg in enumerate(algorithms):  # on each algorithm
                    print('running...dataset[{}/{}] experiment[{}/{}]({}x{}) algorithm[{}/{}] for[{}/{}] '.format(
                        cnt_dataset + 1, cnt_data_sets, cnt_experiment + 1, cnt_experiments, n, p, cnt_alg + 1,
                        cnt_algorithms,
                        i + 1, num_starts), end='')

                    # Construct the algorithm
                    lts = fit_algorithm_exact(alg, intercept=intercept, X=X, y=y, h_size=h_size,
                                              num_starts=probabilistic_starts,
                                              max_steps=probabilistic_steps)

                    iters = lts.n_iter_
                    time = lts.time_total_
                    rss = lts.rss_
                    subset = lts.h_subset_
                    subset.sort()
                    subset = np.asarray(subset)

                    lts_h_size = subset.shape[0]

                    print('t: {0:.2f}'.format(time))

                    res = res.append(pd.Series([alg, n, p, outlier_ratio, outlier_second_model_ratio, rss, iters, time,
                                                intercept, lts_h_size,
                                                max_steps, leverage_ratio], index=res.columns), ignore_index=True)

                    # save experiments to file [prevent loosing data on failure]
                    res.to_csv(output)


def solve_clean(X_clean, y_clean, h_size, intercept):
    X_clean, y_clean = validate(X_clean, y_clean, use_intercept=intercept)
    X_clean = np.asmatrix(X_clean)
    y_clean = np.asmatrix(y_clean)
    q, r = linalg.qr(X_clean)
    p = r.shape[1]
    r1 = r[:p, :]
    qt = q.T
    q1 = qt[:p, :]
    theta = linalg.solve_triangular(r1, q1 * y_clean)  # p x substitution
    residuals = y_clean - X_clean * theta

    # h subset
    residuals = residuals.T
    residuals = np.ravel(residuals)
    sort_args = np.argsort(residuals)
    h_subset = sort_args[:h_size]

    # rss on h subset
    residuals.sort()
    residuals_h = residuals[:h_size]
    rss_clean = np.dot(residuals_h, residuals_h.T)

    if intercept:
        intercept = theta[-1, 0]  # last row first col
        coef = np.ravel(theta[:-1, 0])  # for all but last column,  only first col
    else:
        intercept = 0.0
        coef = np.ravel(theta[:, 0])  # all rows, only first col

    return coef, intercept, rss_clean, h_subset


# currently support for np.ndarray and matrix
def validate(X, y, use_intercept):
    if X.ndim == 1:
        X = np.reshape(X, [X.shape[0], 1])
    if y.ndim == 1:
        y = np.reshape(y, [y.shape[0], 1])

    if type(X) is not np.ndarray:
        X = np.ndarray(X)
    if type(y) is not np.ndarray:
        y = np.ndarray(y)

    if use_intercept:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    return X, y
