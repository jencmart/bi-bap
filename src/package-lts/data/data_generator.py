import numpy as np
import math

# Data generator
def generate_data_2D_multi_variate(cnt, outlier_percentage=20):
    # LINEAR DATA
    # data generated same way as in Rousseeuw and Driessen 2000
    N_clean = cnt - int(math.floor(cnt/100*outlier_percentage))
    N_dirty = int(math.ceil(cnt/100*outlier_percentage))

    X_original = np.random.normal(loc=0, scale=10, size=N_clean)  # var = 100
    e = np.random.normal(loc=0, scale=1, size=N_clean)  # var = 1
    y_original = 1 + X_original + e
    # OUTLIERS
    # multivariate N(mean = location, covariance)
    # diagonalni 25 I
    outliers = np.random.multivariate_normal(mean=[50, 0],
                                             cov=[[25, 0], [0, 25]],
                                             size=N_dirty)

    # outliers
    # FINAL DATA
    X = np.concatenate((X_original, outliers.T[0]), axis=0)
    y = np.concatenate((y_original, outliers.T[1]), axis=0)

    return X, y


def generate_data_ND(cnt, dim, outlier_percentage=20, n_xij= (0,10), ei = (0, 1), n_xi1_outlier = (100,10) ):
    N_clean = cnt - int(math.floor(cnt / 100 * outlier_percentage))
    N_dirty = int(math.ceil(cnt / 100 * outlier_percentage))

    # Xij
    mu, sigma = n_xij
    X_clean = np.random.normal(mu,sigma, (N_clean, dim))
    mu, sigma = n_xi1_outlier
    X_outliers = np.random.normal(mu,sigma, (N_dirty, dim))

    #ei
    mu, sigma = ei
    e = np.random.normal(mu, sigma, (N_clean, 1))
    e_2 = np.random.normal(mu, sigma, (N_dirty, 1))

    #Y
    y_clean = np.concatenate((X_clean, e), axis=1)
    y_clean = np.sum(y_clean, axis=1)

    y_outliers = np.concatenate((X_outliers, e_2), axis=1)
    y_outliers = np.sum(y_outliers, axis=1)

    X = np.concatenate((X_clean, X_outliers), axis=0)
    y = np.concatenate((y_clean, y_outliers), axis=0)

    y = np.reshape(y, [y.shape[0], 1])
    y_clean = np.reshape(y_clean, [y_clean.shape[0], 1])

    return X, y, X_clean, y_clean


# generate y (can produce vertical outliers)
# normal, exponential, lognormal
def generate_y(X, mu, sigma, distribution='n'):
    # SD (s) := standard deviation := sigma := scale   Standard deviation (spread or “width”) of the distribution.
    # E := expected value ~ mean := mu      := loc     Mean (“centre”) of the distribution.

    if (distribution == 'n'):
        e = np.random.normal(mu, sigma, (X.shape[0], 1))
    elif (distribution == 'ln'):
        e = np.random.lognormal(mu, sigma, (X.shape[0], 1))
    else:
        e = np.random.exponential(sigma, (X.shape[0], 1))

    y = np.concatenate((X, e), axis=1)
    y = np.sum(y, axis=1)
    return y


def generate_dataset(n, p,  # X \in R^{n x p}
                     outlier_ratio=0.3,  # ratio of the outliers in the whole data set
                     leverage_ratio=0.1,  # ratio of data outlying in x , in the whole dataset
                     x_ms=(0, 10),  # not outlying x  ~ N(mean, sd)
                     x_lav_ms=(100, 10),  # outlying x  ~ N(mean, sd)
                     e_ms=(0, 1),  # not outlying y  e ~ N(mean, sd)
                     e_out_ms=(-1, 400),  # outlying y   e ~ N(mean, std)  or  ~Exp(std)   'n'   'e'
                     e_out_dist='n',  # n/ln/e distribution of e for outying y
                     outlier_secon_model_ratio=0.5, # ratio of outlers which are not outling in (y) but instead are form comletely different model  (if 0, data only from one model)
                     coeff_scale=50, # random vector of regression coefficients c \in { (-coeff_scale, coeff_cale)^p } \ 0  , so that yi = c xi.T + e
                     mod2_x_ms=(0, 10),  # second model  x~N(m,s)
                     mod2_e_ms=(0, 1)):  # second model e~N(m,s)

    # Calculate number of elements in each catogry (probability tree diagram)

    # depth 1
    N_clean = int(math.floor(n * (1 - outlier_ratio)))
    N_dirty = n - N_clean

    # depth 2
    _N_clean_lav = int(math.floor(N_clean * leverage_ratio))
    _N_clean_non_lav = N_clean - _N_clean_lav

    # depth 2
    N_dirty_lav = int(math.floor(N_dirty * leverage_ratio))
    N_dirty_non_lav = N_dirty - N_dirty_lav

    # depth 3
    _N_dirt_lav_y = int(math.floor(N_dirty_lav * (1 - outlier_secon_model_ratio)))
    _N_dirt_lav_model2 = N_dirty_lav - _N_dirt_lav_y

    # depth 3
    _N_dirty_non_lav_y = int(math.floor(N_dirty_non_lav * (1 - outlier_secon_model_ratio)))
    _N_dirty_non_lav_model2 = N_dirty_non_lav - _N_dirty_non_lav_y

    # GENERATE X

    # X clean && laverage
    mu, sigma = x_lav_ms
    X_clean_lav = np.random.normal(mu, sigma, (_N_clean_lav, p))

    # X clean && non laverage
    mu, sigma = x_ms
    X_clean_non_lav = np.random.normal(mu, sigma, (_N_clean_non_lav, p))

    # X dirty && laverage && y_out
    mu, sigma = x_lav_ms
    X_dirty_lav_y = np.random.normal(mu, sigma, (_N_dirt_lav_y, p))

    # X dirty && laverage && model2
    mu, sigma = x_lav_ms
    X_dirty_lav_model2 = np.random.normal(mu, sigma, (_N_dirt_lav_model2, p))

    # X dirty && non laverage && y_out
    mu, sigma = x_ms
    X_dirty_non_lav_y = np.random.normal(mu, sigma, (_N_dirty_non_lav_y, p))

    # X dirty && non laverage && model2
    # mu, sigma = x_ms - todo zmena
    mu, sigma = mod2_x_ms
    X_dirty_non_lav_model2 = np.random.normal(mu, sigma, (_N_dirty_non_lav_model2, p))

    # CONCATENATE clean data, outliers 1.type and outliers 2.type
    X_clean = np.concatenate((X_clean_lav, X_clean_non_lav), axis=0)
    X_dirty_y = np.concatenate((X_dirty_lav_y, X_dirty_non_lav_y), axis=0)
    X_dirty_model2 = np.concatenate((X_dirty_lav_model2, X_dirty_non_lav_model2), axis=0)

    # Create the outliers of 1st type (second model)  (( this can be easily modified to create d different models ... )))

    # first, generate random coefficients
    coefficients = np.random.randint(low=1, high=coeff_scale, size=p)
    for i in coefficients.shape:
        if np.random.rand() >= 0.5:
            coefficients[i - 1] = -1 * coefficients[i - 1]

    # second, multiply columns of X with those coefficients
    X_dirty_model2_tmp = np.multiply(coefficients, X_dirty_model2)

    # GENERATE Y

    # Y clean
    mu, sigma = e_ms
    y_clean = generate_y(X_clean, mu=mu, sigma=sigma, distribution='n')

    # Y dirty_y
    mu, sigma = e_out_ms
    y_dirty_y = generate_y(X_dirty_y, mu=mu, sigma=sigma, distribution=e_out_dist)

    # Y dirty(clean) second model
    # mu, sigma = e_ms   todo -zmena
    mu, sigma = mod2_e_ms
    y_dirty_model2 = generate_y(X_dirty_model2_tmp, mu=mu, sigma=sigma, distribution='n')

    # Concatenate DATASET
    X = np.concatenate((X_clean, X_dirty_y, X_dirty_model2), axis=0)
    y = np.concatenate((y_clean, y_dirty_y, y_dirty_model2), axis=0)
    y = np.reshape(y, [y.shape[0], 1])
    y_clean = np.reshape(y_clean, [y_clean.shape[0], 1])

    # Shuffle rows of dirty data...
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]

    return X, y, X_clean, y_clean
