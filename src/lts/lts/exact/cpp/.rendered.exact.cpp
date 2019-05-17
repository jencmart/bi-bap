/*cppimport

*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/LU>
#include <Eigen/QR>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <time.h>

/*
system_clock - real time clock which is actually the clock used by system
high_resolution_clock - clock with the smallest tick/interval available/supported by the system
stable_clock - a clock with a steady tick rate (Recommended as tick rate is stable, while system_clock's tick period varies according to system load, see this
*/

namespace py = pybind11;

struct ResultExact{
    public:
        std::vector<int> hSubset;
        Eigen::MatrixXd theta;
        double rss;
        int n_iter;
        double time1;
        double time2;
        double time3;
        bool converged;
        ResultExact(const std::vector<int> & h_sub, const Eigen::MatrixXd & theta_hat, double RSS, int n_iterations): hSubset(h_sub){
            theta = theta_hat;
            rss = RSS;
            n_iter = n_iterations;
            converged = false;
            time1 = 0;
            time2 = 0;
            time3 = 0;
        }

        double getRSS(){
            return rss;
        }
        double getTime1(){
            return time1;
        }
        double getTime2(){
            return time2;
        }
        double getTime3(){
            return time3;
        }
        Eigen::MatrixXd getTheta(){
            return theta;
        }
        std::vector<int> getHSubset(){
        return hSubset;
        }
        int getNIter(){
           return n_iter;
        }
};



// *********************************************************************************************************************
// ***********************   S U B R O U T I N E S   *******************************************************************
// *********************************************************************************************************************

// subroutine to calculate inner product x M x.T, where M is orthogonal matrix
double idx_inv_idx(const Eigen::MatrixXd & dataMatrix, int idx, const Eigen::MatrixXd & inversion){
    double xmx =  ( dataMatrix(idx, Eigen::all) * inversion * (dataMatrix(idx, Eigen::all).transpose()))(0,0);
    return xmx;
}

// subroutine to calculate theta using inversion; theta = (X.T X)^{-1} X.T y
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> theta_and_inversion(const Eigen::MatrixXd & dataJX,
                                        const Eigen::MatrixXd & dataJy){
    // inversion =  (xT X).I ; theta = inversion * x.T * y ; residuals = all y - all * theta
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomposition(dataJX.transpose() * dataJX);
    Eigen::MatrixXd inversion = qr_decomposition.inverse();
    Eigen::MatrixXd theta = inversion * dataJX.transpose() * dataJy  ; // pxp * pxh  * hx1

    return  std::make_tuple(theta, inversion);
}

// subroutine to update theta and inversion if row is included to X (Agullo)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> theta_and_inversion_plus(const Eigen::MatrixXd & theta,
                                        const Eigen::MatrixXd &  dataMX, const Eigen::MatrixXd &  dataMy, int jSwap,
                                        const Eigen::MatrixXd &  inversion){
    double imi =  idx_inv_idx(dataMX, jSwap, inversion);
    Eigen::MatrixXd xi = dataMX(jSwap, Eigen::all);
    Eigen::MatrixXd yi = dataMy(jSwap, Eigen::all);

    double w = -1 / (1 + imi);
    Eigen::MatrixXd u = inversion * (xi.transpose());
    Eigen::MatrixXd theta_plus = theta + (-1 * (yi - (xi*theta))(0, 0) * (w*u)); // O(p)  # !!!! (changed [* -1] )
    Eigen::MatrixXd inversion_plus = inversion + (w * u * u.transpose());   // O(p^2)

    return std::make_tuple(theta_plus, inversion_plus);
}

// calculate increase of the RSS if observation is inserted to the matrix (Agullo)
double gamma_plus_inv(const Eigen::MatrixXd & dataMX,
                                const Eigen::MatrixXd & dataMy,
                                int j,
                                const Eigen::MatrixXd & inversion,
                                const Eigen::MatrixXd & theta){

    // calculate gamma+ O(p^2)
    double imi = idx_inv_idx(dataMX, j, inversion);
    Eigen::MatrixXd xi = dataMX(j, Eigen::all);
    Eigen::MatrixXd yi = dataMy(j, Eigen::all);
    double gamma_plus = std::pow((yi - xi*theta)(0, 0), 2) / (1 + imi);

    return gamma_plus;
}


std::tuple<Eigen::MatrixXd, double> thetaAndRss(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, const std::vector<int> & indexes)
{
    // Create the sub-matrices
    Eigen::MatrixXd dataX = X(indexes, Eigen::all);
    Eigen::MatrixXd dataY = y(indexes, Eigen::all);

    // solve OLS
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(dataX);
    Eigen::MatrixXd theta = qr.solve(dataY);
    Eigen::MatrixXd residuals = dataY - dataX * theta;
    double rss = (residuals.transpose() * residuals )(0,0);

    return  std::make_tuple(theta, rss);
}

template <typename T>
std::vector<int> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}


#define Abs(x)    ((x) < 0 ? -(x) : (x))
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define EPSILON 0.000000001

double relDif(double a, double b)
{
	double c = Abs(a);
	double d = Abs(b);

	d = Max(c, d);

	return d == 0.0 ? 0.0 : Abs(a - b) / d;
}

// *********************************************************************************************************************
// ************************   E X H A U S T I V E   ********************************************************************
// *********************************************************************************************************************
ResultExact * refinementExhaustive(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize ) {

    double rss_min = std::numeric_limits<double>::infinity();
    std::vector<int> indexes_min;

    // bool vector for the combinations
    int n = X.rows();
    std::vector<bool> bool_indexes(n);
    std::fill(bool_indexes.begin(), bool_indexes.begin() + hSize, true);

    do {
        std::vector<int> indexes;

        // Create index array
        for (int i = 0; i < n; ++i)
            if (bool_indexes[i])
                indexes.push_back(i);

        // Solve OLS on the subset
        Eigen::MatrixXd theta;
        double rss;
        std::tie(theta, rss) = thetaAndRss(X, y, indexes);

        // update if needed
        if(rss < rss_min){
            rss_min = rss;
            indexes_min.swap(indexes);
        }

    } while (std::prev_permutation(bool_indexes.begin(), bool_indexes.end()));


    // solve OLS
    Eigen::MatrixXd theta;
    std::tie(theta, rss_min) = thetaAndRss(X, y, indexes_min);
    return new ResultExact(indexes_min, theta, rss_min, 0);
}


// *********************************************************************************************************************
// ************************   B R A N C H   A N D   B O U N D   ********************************************************
// *********************************************************************************************************************

// combinatorial tree traversal (RTL post-order)
// a,b needs to be copied (we are saving the path)
// theta, inversion also copied (path...)
// dept copied (path..)
// rss copied (path..)
// rss_min && indexes_min && cuts ... REF !
void traverse_recursive(const Eigen::MatrixXd & X,
                        const Eigen::MatrixXd & y,
                        std::vector<int> a,
                        std::vector<int> b,
                        int depth,
                        double rss,
                        Eigen::MatrixXd theta,
                        Eigen::MatrixXd inversion,
                        double & rss_min,
                        std::vector<int> & indexes_min,
                        int & cuts,
                        int & hSize){

    double rss_here = 0.0;
    Eigen::MatrixXd theta_here, inversion_here;

    // bottom of the tree (leaf)
    if(depth == hSize){

        // calculate gama plus and new RSS
        double gamma_plus = gamma_plus_inv(X, y, a.back(), inversion, theta);
        rss_here = rss + gamma_plus;

        if(rss_here < rss_min){
            rss_min = rss_here;
            indexes_min = a;
        }
        return;
    }

    // most common case - depth > p but not leaf
    if((int)a.size() > X.cols()){
        // calculate gama plus and new RSS
        double gamma_plus = gamma_plus_inv(X, y, a.back(), inversion, theta);
        rss_here = rss + gamma_plus;

        // bounding condition
        if(rss_here >= rss_min){
            cuts +=1;
            return;
        }

        // update theta and inversion
        std::tie(theta_here, inversion_here) = theta_and_inversion_plus(theta, X, y, a.back(), inversion);
    }
    // depth p - "root" - calculate theta and inversion for the first time
    else if((int)a.size() == X.cols()){

        // Create the sub-matrices
        Eigen::MatrixXd dataX = X(a, Eigen::all);
        Eigen::MatrixXd dataY = y(a, Eigen::all);

        // And calculate theta and inversion and RSS
        std::tie(theta_here, inversion_here) = theta_and_inversion(dataX, dataY);
        Eigen::MatrixXd residuals = dataY - dataX * theta_here;
        rss_here = (residuals.transpose() * residuals )(0,0);
    }

    // till we can go deeper
    while(! b.empty()){

        // not enough to produce h subset in ancestors
        if((int)a.size() + (int)b.size() < hSize)
            break;

        // move element from a to b
        a.push_back(b.back());
        b.pop_back();

        // go deeper
        traverse_recursive(X, y, a, b, depth +1, rss_here, theta_here, inversion_here, rss_min, indexes_min, cuts, hSize);

        // remove from a
        a.pop_back();
    }
}



ResultExact * refinementBab(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize) {

    // init variables
    double rss_min = std::numeric_limits<double>::infinity();
    std::vector<int> indexes_min;
    int cuts = 0;

    // prepare index vectors
    unsigned N = X.rows();
    std::vector<int> a;
    std::vector<int> b(N);
    std::iota(b.begin(), b.end(), 0);


    Eigen::MatrixXd theta_tmp;
    Eigen::MatrixXd inversion_tmp;
    double rss_tmp = -1;
    double depth_tmp = 0;
    traverse_recursive(X, y, a, b, depth_tmp, rss_tmp, theta_tmp, inversion_tmp, rss_min, indexes_min, cuts, hSize);

    // Create the sub-matrices
    Eigen::MatrixXd dataX = X(indexes_min, Eigen::all);
    Eigen::MatrixXd dataY = y(indexes_min, Eigen::all);

    // solve OLS
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(dataX);
    Eigen::MatrixXd theta_final = qr.solve(dataY);

    // not needed actually... we have rss_min, but this improves num. stability
    Eigen::MatrixXd residuals = dataY - dataX * theta_final;
    double rss = (residuals.transpose() * residuals )(0,0);
    return new ResultExact(indexes_min, theta_final, rss, cuts);
}









// *********************************************************************************************************************
// ************************   B O R D E R    S C A N N I N G   *********************************************************
// *********************************************************************************************************************

void all_h_subsets_bsa(const std::vector<double> & vecResiduals,
                       const std::vector<int> & sort_args,
                       int p,
                       double rss_min,
                       std::vector<std::vector<int>> & all_subsets,
                       int hSize,
                       const Eigen::MatrixXd & X,
                       const Eigen::MatrixXd & y,
                       int & cuts,
                       bool first_set){

    double res_h =  vecResiduals[sort_args[hSize- 1]];  // r_h residuum

    // find smallest index i; i <= h;  so that r_i = r_h
    int idx_i = hSize - 1;
    for(int i = idx_i; i >= 0 ; i-- )
        if(relDif( vecResiduals[sort_args[i]], res_h) <= EPSILON)  // if at idx_i == res_h
            idx_i = i;

    // find greatest index j; j >= h+1 ;  so that r_j = r_{h+1} ; [ h+1 because we know that r_h == r_{h+1} ]
    int idx_j = hSize;
    for(int j = hSize; j < (int)vecResiduals.size() ; j++ )
        if(relDif( vecResiduals[sort_args[j]], res_h) <= EPSILON) // if at idx_i == res_h
            idx_j = j;

    // BSA-BAB speedup
    int cntUnique =  idx_i; // idx_i - 1 + 1 (- because at idx_i is smallest r_i = r_h) (+ because it is index)
    if(cntUnique >= p){
        std::vector<int> begin(sort_args.begin(), sort_args.begin() + cntUnique);
        // Solve OLS on the subset
        Eigen::MatrixXd theta;
        double rss;
        std::tie(theta, rss) = thetaAndRss(X, y, begin);

        if(rss > rss_min){
            cuts += 1;
            return;
        }

        if(relDif(rss, rss_min) <= EPSILON && first_set) {
            cuts += 1;
            return;
        }

    }

    // create all combinations of size:= self._h_size-1 - idx_i + 1
    // #equal residuals from i to h included (h-i+1)
    // we are using indexes so (h-1-i+1) ... h - idx_i where h is #resuidals
    int length = idx_j - idx_i + 1;
    std::vector<bool> bool_indexes(length);
    std::fill(bool_indexes.begin(), bool_indexes.begin() + hSize - idx_i, true); // we want  hSize - idx_i subsets (ones)

    // and append rest of the indexes for each combination
    do {
        std::vector<int> indexes;

        // Create index array --> create combinations
        for (int i = 0; i < length; ++i)
            if (bool_indexes[i])
                indexes.push_back(i + idx_i); // we want combinations starting form idx_i

        // start with first i unique indexes [0, 1, ... i-1]
        std::vector<int> begin(sort_args.begin(), sort_args.begin() + cntUnique);

        // append the rest of the begin with one combination
        for(auto idx : indexes)
            begin.push_back(sort_args[idx]);

        // push it to the vector of all combs
        all_subsets.push_back(begin);

    } while (std::prev_permutation(bool_indexes.begin(), bool_indexes.end()));
}



// ***************************************************************************************************
// ********************************* RANDOM BSA ******************************************************
// ***************************************************************************************************

ResultExact * refinementRandomBsa(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize, int max_subsets) {

    // init variables
    double rss_min = std::numeric_limits<double>::infinity();

    bool first_set = false;

    std::vector<int> indexes_min;
    Eigen::MatrixXd theta_min;
    int cuts = 0;

    // bool vector for the combinations
    int p = X.cols();
    int N = X.rows();


    // fit some naive start
    std::vector<int> naive(hSize);
    std::iota(naive.begin(), naive.end(), 0);
    Eigen::MatrixXd naive_theta;
    double naive_rss;
    std::tie(naive_theta, naive_rss) = thetaAndRss(X, y, naive);
    rss_min = naive_rss;
    first_set = true;
    indexes_min = naive;
    theta_min = naive_theta;

    for(int r = 0; r < max_subsets ; r++) {

        // Generate random p+1 subset
        std::vector<int> permutation(N);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::random_shuffle(permutation.begin(), permutation.end());
        //  and slice first p +1 elements of this permutation
        std::vector<int> indexes(permutation.begin(), permutation.begin() + p+1);


        // store one element by side
        int first_idx = indexes.back();
        indexes.pop_back();
        Eigen::MatrixXd x1 = X(first_idx, Eigen::all);
        Eigen::MatrixXd y1 = y(first_idx, Eigen::all);


        // for all sign counts
        for(int i = 1; i <= p; ++i){

            // vectors of length p
            std::vector<bool> sign_indexes(p);
            // containing i ones an p-1 zeros  (1 represents +, 0 represents -
            std::fill(sign_indexes.begin(), sign_indexes.begin() + i, true);
            do {

                for(int j= 0 ; j < (int)sign_indexes.size(); j++)
                {
                    // Create the sub-matrices form (p) indexes
                    Eigen::MatrixXd dataX = X(indexes, Eigen::all);
                    Eigen::MatrixXd dataY = y(indexes, Eigen::all);

                    // create the equations
                    for(int row_idx = 0; row_idx < dataX.rows() ; row_idx++){
                        if(j){
                            dataX(row_idx, Eigen::all) = x1 - dataX(row_idx, Eigen::all);
                            dataY(row_idx, Eigen::all) = y1 - dataY(row_idx, Eigen::all);
                        }else{
                        dataX(row_idx, Eigen::all) = x1 + dataX(row_idx, Eigen::all);
                        dataY(row_idx, Eigen::all) = y1 + dataY(row_idx, Eigen::all);
                        }
                    }

                    // solve theta
                    Eigen::HouseholderQR<Eigen::MatrixXd> qr(dataX);
                    Eigen::MatrixXd theta = qr.solve(dataY);

                    // calculate all residuals, square them and sort them
                    Eigen::MatrixXd residuals = y - X * theta;
                    residuals = residuals.transpose(); // squared 1 x n
                    std::vector<double> vecResiduals(residuals.data(), residuals.data() + residuals.cols());
                    for(unsigned k = 0; k < residuals.size(); k++)
                        vecResiduals[k] = std::pow(vecResiduals[k], 2);

                    std::vector<int> sort_args =  sort_indexes(vecResiduals);


                    // calculate xi1 residuum, r_h and r_{h+1}
                    double x1_res = std::pow( (y1 - x1*theta)(0, 0) , 2);
                    double res_h =  vecResiduals[sort_args[hSize - 1]];  // r_h residuum
                    double res_h_1 = vecResiduals[sort_args[hSize]];    // r_{h+1} residuum


                    // Find all subsets in relation with theta
                    std::vector<std::vector<int>> all_subsets;

                    if(relDif(x1_res, res_h) <= EPSILON){
                        if(relDif(res_h, res_h_1) <=  EPSILON){
                            all_h_subsets_bsa(vecResiduals, sort_args, p, rss_min, all_subsets, hSize, X, y, cuts, first_set);
                        }else{
                            sort_args.resize(hSize);
                            all_subsets.push_back(sort_args);
                        }

                        // for each subset calculate OLS and eventually update
                        for( auto subset : all_subsets){

                            // Solve OLS on the subset
                            Eigen::MatrixXd theta;
                            double rss;
                            std::tie(theta, rss) = thetaAndRss(X, y, subset);

                            if(rss < rss_min){
                                first_set = true;
                                rss_min = rss;
                                indexes_min = subset;
                                theta_min = theta;
                            }
                        }
                    }
                }
            } while (std::prev_permutation(sign_indexes.begin(), sign_indexes.end()));
        }
    }

    return new ResultExact(indexes_min, theta_min, rss_min, cuts);
}





// ***************************************************************************************************
// *********************************  BSA  ***********************************************************
// ***************************************************************************************************
ResultExact * refinementBsa(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize, double set_rss) {

    // init variables
    double rss_min;
    if(set_rss < 0)
        rss_min = std::numeric_limits<double>::infinity();
    else
        rss_min = set_rss;

    bool first_set = false;

    std::vector<int> indexes_min;
    Eigen::MatrixXd theta_min;
    int cuts = 0;

    // bool vector for the combinations
    int p = X.cols();
    int N = X.rows();
    std::vector<bool> bool_indexes(N);
    std::fill(bool_indexes.begin(), bool_indexes.begin() + p+1, true); // we want  p+1 subsets (ones)
    do {
        std::vector<int> indexes;

        // Create index array
        for (int i = 0; i < N; ++i)
            if (bool_indexes[i])
                indexes.push_back(i);

        // store one element by side
        int first_idx = indexes.back();
        indexes.pop_back();
        Eigen::MatrixXd x1 = X(first_idx, Eigen::all);
        Eigen::MatrixXd y1 = y(first_idx, Eigen::all);


        // for all sign counts
        for(int i = 1; i <= p; ++i){

            // vectors of length p
            std::vector<bool> sign_indexes(p);
            // containing i ones an p-1 zeros  (1 represents +, 0 represents -
            std::fill(sign_indexes.begin(), sign_indexes.begin() + i, true);
            do {

                for(int j= 0 ; j < (int)sign_indexes.size(); j++)
                {
                    // Create the sub-matrices form (p) indexes
                    Eigen::MatrixXd dataX = X(indexes, Eigen::all);
                    Eigen::MatrixXd dataY = y(indexes, Eigen::all);

                    // create the equations
                    for(int row_idx = 0; row_idx < dataX.rows() ; row_idx++){
                        if(j){
                            dataX(row_idx, Eigen::all) = x1 - dataX(row_idx, Eigen::all);
                            dataY(row_idx, Eigen::all) = y1 - dataY(row_idx, Eigen::all);
                        }else{
                        dataX(row_idx, Eigen::all) = x1 + dataX(row_idx, Eigen::all);
                        dataY(row_idx, Eigen::all) = y1 + dataY(row_idx, Eigen::all);
                        }
                    }

                    // solve theta
                    Eigen::HouseholderQR<Eigen::MatrixXd> qr(dataX);
                    Eigen::MatrixXd theta = qr.solve(dataY);

                    // calculate all residuals, square them and sort them
                    Eigen::MatrixXd residuals = y - X * theta;
                    residuals = residuals.transpose(); // squared 1 x n
                    std::vector<double> vecResiduals(residuals.data(), residuals.data() + residuals.cols());
                    for(unsigned k = 0; k < residuals.size(); k++)
                        vecResiduals[k] = std::pow(vecResiduals[k], 2);

                    std::vector<int> sort_args =  sort_indexes(vecResiduals);


                    // calculate xi1 residuum, r_h and r_{h+1}
                    double x1_res = std::pow( (y1 - x1*theta)(0, 0) , 2);
                    double res_h =  vecResiduals[sort_args[hSize - 1]];  // r_h residuum
                    double res_h_1 = vecResiduals[sort_args[hSize]];    // r_{h+1} residuum


                    // Find all subsets in relation with theta
                    std::vector<std::vector<int>> all_subsets;

                    if(relDif(x1_res, res_h) <= EPSILON){
                        if(relDif(res_h, res_h_1) <=  EPSILON){
                            all_h_subsets_bsa(vecResiduals, sort_args, p, rss_min, all_subsets, hSize, X, y, cuts, first_set);
                        }else{
                            sort_args.resize(hSize);
                            all_subsets.push_back(sort_args);
                        }

                        // for each subset calculate OLS and eventually update
                        for( auto subset : all_subsets){

                            // Solve OLS on the subset
                            Eigen::MatrixXd theta;
                            double rss;
                            std::tie(theta, rss) = thetaAndRss(X, y, subset);

                            if(rss < rss_min){
                                first_set = true;
                                rss_min = rss;
                                indexes_min = subset;
                                theta_min = theta;
                            }
                        }
                    }
                }
            } while (std::prev_permutation(sign_indexes.begin(), sign_indexes.end()));
        }
    } while (std::prev_permutation(bool_indexes.begin(), bool_indexes.end()));

    return new ResultExact(indexes_min, theta_min, rss_min, cuts);
}


// ***********************************************************************************
// ***************************  RANDOM EXHAUSTIVE   *******************************************************
// *********************************************************************************


ResultExact * refinementExhaustiveRandom(const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize, int max_subsets ) {

    double rss_min = std::numeric_limits<double>::infinity();
    std::vector<int> indexes_min;


   // bool vector for the combinations
    // int p = X.cols();
    int N = X.rows();

    for(int r = 0; r < max_subsets ; r++) {

            // Generate random p+1 subset
            std::vector<int> permutation(N);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::random_shuffle(permutation.begin(), permutation.end());
            //  and slice first p +1 elements of this permutation
            std::vector<int> indexes(permutation.begin(), permutation.begin() + hSize);


        // Solve OLS on the subset
        Eigen::MatrixXd theta;
        double rss;
        std::tie(theta, rss) = thetaAndRss(X, y, indexes);

        // update if needed
        if(rss < rss_min){
            rss_min = rss;
            indexes_min.swap(indexes);
        }

    }


    // solve OLS
    Eigen::MatrixXd theta;
    std::tie(theta, rss_min) = thetaAndRss(X, y, indexes_min);
    return new ResultExact(indexes_min, theta, rss_min, 0);
}



// *********************************************************************************************************************
// *********************  E X A C T  -  S O L U T I O N   **************************************************************
// *********************************************************************************************************************
ResultExact* exact_lts(Eigen::MatrixXd X, Eigen::MatrixXd y, int hSize, int alg, int calc,
                                  double rss, int max_subsets) {
    std::vector<ResultExact*> subsetResults;
    clock_t t;
    t = clock();

    ResultExact * result;

    if(alg == 0){
        // exact exhaustive algorithm
        result = refinementExhaustive(X, y, hSize);

    }else if(alg == 1){
        // exact branch and bound algorithm
        result = refinementBab(X, y, hSize);
    }else if (alg == 2) {
        // exact border scanning algorithm
        result = refinementBsa(X, y, hSize, rss);
    }else if (alg == 3){
        result = refinementRandomBsa(X, y, hSize, max_subsets);
    } else {
        result = refinementExhaustiveRandom(X, y, hSize, max_subsets);
    }

    // append result to result array
    subsetResults.push_back(result);

    float time1 = ((float)(clock() - t))/CLOCKS_PER_SEC;

    // find and return best results
    ResultExact* best = subsetResults[0];
//    for (int i = 0 ; i < numStarts; ++i)
//        best = subsetResults[i]->rss < best->rss ? subsetResults[i] : best;
    // save total time (performance statistics)
    best->time1 = time1;

    return best;
}


// *********************************************************************************************************************
// *********************************************************************************************************************
// **********************    P  Y  B  I  N  D  1  1   ******************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// FOR THE PYBIND11
// LEARN ABOUT py::vectorize()

// # To work way you expect, you must adhere the naming convention right:
// xxx.cpp
//	PYBIND11_MODULE(xxx, m)

// test.py
//	my_import =  cppimport.imp("xxx")

PYBIND11_MODULE(exact, m) {

	// documentation - not necessary
	 m.doc() = R"pbdoc(
		    Pybind11 exact solution of the lts estimate
		    -----------------------

		    .. currentmodule:: exact

		    .. autosummary::
		       :toctree: _generate

		       exact_algorithms
		       implementations of following algorithms:
		       exhaustive, branch-and-bound, border-scanning-algorithm
		)pbdoc";


    // bind functions - and use documentation
    m.def("exact_lts", &exact_lts,  R"pbdoc(
        Exact algorithms calculating LTS estimate
    )pbdoc");

    // bind classes -- warning !!! -- same class name (only in c++) leads to "generic type ClassName already registered"
    py::class_<ResultExact>(m, "ResultExact")
    .def(py::init< const std::vector<int> & , const Eigen::MatrixXd & , double ,int  >())
    .def("get_rss", &ResultExact::getRSS)
    .def("get_theta", &ResultExact::getTheta)
    .def("get_h_subset", &ResultExact::getHSubset)
    .def("get_n_inter", &ResultExact::getNIter)
    .def("get_time_1", &ResultExact::getTime1, "total time")
    .def("get_time_2", &ResultExact::getTime2, "unused")
    .def("get_time_3", &ResultExact::getTime3, "unused")
    ;

// version info - not neccesary
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
