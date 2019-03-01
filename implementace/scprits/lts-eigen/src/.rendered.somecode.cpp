/*

*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/LU>
#include <Eigen/QR>
#include <iostream>
#include <vector>
#include <algorithm>

namespace py = pybind11;




class Results {
    public:
        Eigen::MatrixXd theta;
        std::vector<int> h_subset;
        double rss;
        int n_iter;

        Results(){
            rss = 0.0;
            n_iter = 0;
        }

};



/* E I G E N */
// convenient matrix indexing comes for free
double get(Eigen::MatrixXd xs, int i, int j) {
    return xs(i, j);
}

// takes numpy array as input and returns double
double det(Eigen::MatrixXd xs) {
    return xs.determinant();
}

// takes numpy array as input and returns another numpy array
Eigen::MatrixXd inv(Eigen::MatrixXd xs) {
    return xs.inverse();
}



void generateSubsets(std::vector<Results*> & subset_results, const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int numStarts, int hSize ) {

    unsigned p = X.cols();
    unsigned N = X.rows();

    for(int i = 0; i < numStarts; ++i){

         // generate random permutation of [pi(0) ... pi(N)]
         std::vector<int> permutation(N);
         std::iota(permutation.begin(), permutation.end(), 0);
         std::random_shuffle ( permutation.begin(), permutation.end());

         // select first p elements of this permutation
         std::vector<int>::const_iterator first = permutation.begin();
         std::vector<int>::const_iterator last  = permutation.begin() + p;
         std::vector<int> ind(first, last);

         // select p samples (at those p random indexes) and calculate matrix decomp
         // LU decompostion in this case
         // todo QR decomposition .. in eigen JacobiRotation
         // FullPivLU worked as expected. lets try QR
         // ColPivHouseholderQR

         Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp(X(ind, Eigen::all));
         // and reveal rank from the decompositon
         auto rank = qr_decomp.rank();

         // add samples till rank is equal p
         while(rank < p && ind.size() < N) {
            ind.push_back(permutation[ind.size()]);
             qr_decomp = qr_decomp.compute(X(ind, Eigen::all)); // todo i am sure this''ll segfault
             rank = qr_decomp.rank();
         }

        //  2x1 :-)
         Eigen::MatrixXd resultat = qr_decomp.solve(y(ind, Eigen::all));
         std::cout << "i am king " << resultat.rows() << 'x' << resultat.cols() << std::endl;
         std::cout << resultat << std::endl;
         //subset_results.push_back(new Results(, ) )); TODO
    }

    return;
}

// return array of weights
void fast_lts(Eigen::MatrixXd X, Eigen::MatrixXd y, int numStarts, int hSize) {

    std::vector<Results*> subset_results;

    generateSubsets(subset_results, X, y, numStarts, hSize);

    return;
}

/* ! E I G E N */


// funkcni jendoducha fce
int square(double x) {
    return x * x;
}




// FOR THE PYBIND11
// LEARN ABOUT py::vectorize()

// m.def .. definuje funkci
// m.def('square', &square)
// to znamena reference na funkci square o par radku vyse
PYBIND11_MODULE(somecode, m) {
    m.def("square", &square);
    m.def("inv", &inv);
    m.def("det", &det);
    m.def("fast_lts", &fast_lts);
}
