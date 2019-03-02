/*
<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['/home/jencmart/bi-bap/implementace/scprits/eigen-lib']

setup_pybind11(cfg)
%>
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

namespace py = pybind11;

struct Result {
    public:
        std::vector<int> hSubset;
        Eigen::MatrixXd theta;
        double rss;
        int n_iter;

        Result(const std::vector<int> & h_sub, const Eigen::MatrixXd & theta_hat, double RSS, int n_iterations): hSubset(h_sub){
            theta = theta_hat;
            rss = RSS;
            n_iter = n_iterations;
        }

        double getRSS(){
            return rss;
        }
};


// *********************************************************************************************************************
// ************************   Q U I C K  -  S E L E C T   **************************************************************
// *********************************************************************************************************************
void kth_smallest_recursive_inplace(Eigen::MatrixXd & arr, std::vector<int> & indexes, int left, int right, int k) {
    double pivot = arr(right,0);
    int pos = left;

    // iterate unsorted part
    for (int i = left; i < right; ++i){
        if (arr(i,0) <= pivot){  // if current pos <= pivot, then swap
            double tmp = arr(pos,0);
            arr(pos,0) = arr(i,0);
            arr(i,0) = tmp;
            // swap indexes array also...
            tmp = indexes[pos];
            indexes[pos] = indexes[i];
            indexes[i] = tmp;
            // position is now shifted
            pos += 1;
        }
    }

    // pos is now right
    double tmp = arr(pos,0);
    arr(pos,0) = arr(right,0);
    arr(right,0) =  tmp;
    // dont forget on second array
    tmp = indexes[pos];
    indexes[pos] = indexes[right];
    indexes[right] =  tmp;

    // end
    if (pos - left == k - 1)
        return;

    // kth smallest is in left part
    if (pos - left > k - 1)
        return kth_smallest_recursive_inplace(arr, indexes, left, pos-1, k);

    // kth smallest is in right part
    else
        return kth_smallest_recursive_inplace(arr, indexes, pos+1, right, k - pos + left - 1);
}



// *********************************************************************************************************************
// ******************** KTH SMALLEST RECURSIVE INPLACE  ****************************************************************
// *********************************************************************************************************************
// TODO - i belive this is ok (at least this is not cause of this error right now)
void kth_smallest_recursive_inplace_NoIndex(std::vector<Result*> & subsetResults, int left, int right, int k){
    double pivot = subsetResults[0]->rss; // todo revrite it all to wrap class and implement '<=' and getPivot
    int pos = left;

    // iterate unsorted part
    for (int i = left; i < right; ++i){
        if (subsetResults[i]->rss <= pivot){  // if current pos <= pivot, then swap
            Result* tmp = subsetResults[pos];
            subsetResults[pos] = subsetResults[i];
            subsetResults[i] = tmp;

            // position is now shifted
            pos += 1;
        }
    }

    // pos is now right
    Result* tmp = subsetResults[pos];
    subsetResults[pos] = subsetResults[right];
    subsetResults[right] =  tmp;

    // end
    if (pos - left == k - 1)
        return;

    // kth smallest is in left part
    if (pos - left > k - 1)
        return kth_smallest_recursive_inplace_NoIndex(subsetResults, left, pos-1, k);

    // kth smallest is in right part
    else
        return kth_smallest_recursive_inplace_NoIndex(subsetResults, pos+1, right, k - pos + left - 1);
}


// *********************************************************************************************************************
// *****************  GENERATE ALL H1  *********************************************************************************
// *********************************************************************************************************************
void generateSubsets(std::vector<Result*> & subsetResults, const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int numStarts, int hSize ) {
    unsigned p = X.cols();
    unsigned N = X.rows();

    for(int i = 0; i < numStarts; ++i){
         // generate random permutation of [pi(0) ... pi(N)]
         std::vector<int> permutation(N);
         std::iota(permutation.begin(), permutation.end(), 0);
         std::random_shuffle ( permutation.begin(), permutation.end());

         //  and slice first p elements of this permutation
         std::vector<int> ind(permutation.begin(), permutation.begin() + p);

         // ColPivHouseholderQR, FullPivLU ... todo try JacobiRotations
         Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp(X(ind, Eigen::all));
         // and reveal rank from the decompositon
         auto rank = qr_decomp.rank();

         // add samples till rank is equal p
         while(rank < p && ind.size() < N) {
            ind.push_back(permutation[ind.size()]);
             qr_decomp = qr_decomp.compute(X(ind, Eigen::all)); // todo possible segfault
             rank = qr_decomp.rank();
         }

         // compute OLS on this random subset matrix of rank p
         Eigen::MatrixXd ols_J = qr_decomp.solve(y(ind, Eigen::all));

         // -----------
         // calculate absolute residuals for each data sample from theta Hyperplane
         Eigen::MatrixXd L1_norm = ( y - X * ols_J ).rowwise().lpNorm<1>();

         // Calculate hSize smallest indexes
         std::vector<int> indexes(N);
         std::iota(indexes.begin(), indexes.end(), 0);
         kth_smallest_recursive_inplace(L1_norm, indexes, 0, N-1, hSize);
         // and slice 0 .. hsize-1
         std::vector<int> slicedVec(indexes.begin(), indexes.begin() + hSize);

         // calculate OLS on this smallest h1 subset
         Eigen::ColPivHouseholderQR<Eigen::MatrixXd> finalDecomp(X(slicedVec, Eigen::all));
         Eigen::MatrixXd theta_final = qr_decomp.solve(y(slicedVec, Eigen::all));
         // ------------


         // save H1, OLS, RSS, and #cSteps
         subsetResults.push_back(new Result(slicedVec, theta_final, -1.0, 0) );
    }
    return;
}


// *********************************************************************************************************************
// ****************   PERFORM C STEPS   ********************************************************************************
// *********************************************************************************************************************
void performCStepsInPlace(Result* result,  const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int hSize, int numSteps,  double threshold) {
    for (int i = 0; i < numSteps; ++i){
        // -----------
        // calculate absolute residuals for each data sample from theta Hyperplane
        Eigen::MatrixXd L1_norm = ( y - X * result->theta ).rowwise().lpNorm<1>();
        unsigned N = X.rows();
        // Calculate hSize smallest indexes
        std::vector<int> indexes(N);
        std::iota(indexes.begin(), indexes.end(), 0);
        kth_smallest_recursive_inplace(L1_norm, indexes, 0, N-1, hSize);
        // and slice 0 .. hsize-1
        std::vector<int> slicedVec(indexes.begin(), indexes.begin() + hSize);

        // calculate OLS on this smallest h1 subset
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> finalDecomp(X(slicedVec, Eigen::all));
        Eigen::MatrixXd theta_new = finalDecomp.solve(y(slicedVec, Eigen::all));
        // -----------

        // save theta
        result->theta = theta_new;

        // >>check stopping criterion<<
        if(threshold > 0) {
            double rss_new =  ( (y - X * result->theta).transpose() * (y - X * result->theta) )(0,0);
            if(std::fabs(result->rss - rss_new) < threshold) {
                result->hSubset = slicedVec;
                result->rss = rss_new;
                result->n_iter +=  numSteps;
                result->n_iter += 1; // this step
                return;
            }
        }
        // >>>> /criterion <<<

        // save hSubset in last step
        if(i+1 == numSteps){
            result->hSubset = slicedVec;
        }
    }

    // calculate RSS and update num of iterations
    result->rss =  ( (y - X * result->theta).transpose() * (y - X * result->theta) )(0,0);
    result->n_iter +=  numSteps;
}


// *********************************************************************************************************************
// *********************   F A S T  -  L T S   *************************************************************************
// *********************************************************************************************************************
Result* fast_lts(Eigen::MatrixXd X, Eigen::MatrixXd y, int numStarts, int numInitialCSteps, int numStartsToFinish, int hSize, int maxCSteps, double threshold) {

    std::vector<Result*> subsetResults;

    // generate initial H1_subsets ( #H1_subsets == numStarts)
    generateSubsets(subsetResults, X, y, numStarts, hSize);

    // -- INITIAL
    // numStarts represent range thus we mean to iterate cSteps on all of initial H1 subsets for now
    for (int i = 0; i < numStarts; ++i)
        performCStepsInPlace(subsetResults[i], X, y, hSize, numInitialCSteps, threshold);
    // sort it
    kth_smallest_recursive_inplace_NoIndex(subsetResults, 0, X.rows()-1, numStartsToFinish);

 // find best
    Result* best = subsetResults[0];
    for (int i = 0 ; i < numStartsToFinish; ++i){
        std::cout << "rss je: " << subsetResults[i]->rss << std::endl;

        if(subsetResults[i]->rss < best->rss)
            best = subsetResults[i];
    }
    // -- FINAL
    for (int i = 0; i < numStartsToFinish; ++i)
        performCStepsInPlace(subsetResults[i], X, y, hSize, maxCSteps, threshold);

    unsigned N = X.rows();
    std::cout << std::endl;
      for (int i = numStartsToFinish; i <  N; ++i){
        std::cout << "rss je: " << subsetResults[i]->rss << std::endl;
        if(subsetResults[i]->rss < best->rss)
            best = subsetResults[i];
    }

    return best;
}



// *********************************************************************************************************************
// *********************************************************************************************************************
// **********************    P  Y  B  I  N  D  1  1   ******************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// FOR THE PYBIND11
// LEARN ABOUT py::vectorize()
PYBIND11_MODULE(somecode, m) {

    // bind functions
    m.def("fast_lts", &fast_lts);

    // bind classes
    py::class_<Result>(m, "Result")
    .def(py::init< const std::vector<int> & , const Eigen::MatrixXd & , double ,int  >())
    .def("get_rss", &Result::getRSS)
    ;
   // .def("get_theta", &Result::getTheta)
   // .def("get_h_subset",&Result::getHSubset)
   // ..def("get_h_subset",&Result::getHSubset);
}