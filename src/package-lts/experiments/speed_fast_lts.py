import cppimport
from scipy import spatial

###########################
# To work way you expect, you must adhere the naming convention right:
# xxx.cpp
#	PYBIND11_MODULE(xxx, m)

# something.py
#	my_import =  cppimport.imp("xxx")
###########################

eigen_lts = cppimport.imp("../src/fastlts")



def getRegressor(type):
    if type == 'ltsPython':
        return FastLtsRegression()
    if type == 'ltsCpp':
        return \
            FastLtsEigenRegressor()

if __name__ == '__main__':

    types = ['ltsPython', 'ltsCpp' ]

    # same as in FAST-LTS paper -- it is a benchamrk
    experiments = [(100,2), (100,3), (100,5), (500,2), (500,3), (500,5), (1000,2),
                   (1000,5), (1000,10), (10000,2), (10000,5), (10000,10), (50000,2), (50000,5)]

    # experiments = [(100,2), (100,3)]

    for experiment in experiments:
        n, p = experiment
        X, y, X_clean, y_clean = generate_data_ND(n, p)
        for kind in types:
            lts = getRegressor(kind)


            # lts
            lts.fit(X, y, use_intercept=True)
            weights_correct = lts.coef_

            # print data
            print('{} ({},{})'.format(kind, n, p))
            print('rss: ', lts.rss_)
            print('itr: ', lts.n_iter_)
            print('tim: ', lts.time_total_)

            #ols
            lts.fit(X_clean, y_clean, use_intercept=True, h_size=X_clean.shape[0])
            weights_expected = lts.coef_
            # print('rsO: ', lts.rss_)
            # cos similarity
            result = 1 - spatial.distance.cosine(weights_correct, weights_expected)
            print('cos: ', result)

