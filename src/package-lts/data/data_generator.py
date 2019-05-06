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


def generate_data_ND(cnt, dim, outlier_percentage=20, n_xij= (0,10), ei = (0,1), n_xi1_outlier = (100,10) ):
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

