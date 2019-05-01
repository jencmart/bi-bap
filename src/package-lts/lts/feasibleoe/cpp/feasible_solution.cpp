/*cppimport
<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['../../../lib/eigen']
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
#include <time.h>
/*
system_clock - real time clock which is actually the clock used by system
high_resolution_clock - clock with the smallest tick/interval available/supported by the system
stable_clock - a clock with a steady tick rate (Recommended as tick rate is stable, while system_clock's tick period varies according to system load, see this
*/

namespace py = pybind11;

struct ResultFeasible {
    public:
        std::vector<int> hSubset;
        Eigen::MatrixXd theta;
        double rss;
        int n_iter;
        double time1;
        double time2;
        double time3;
        bool converged;
        ResultFeasible(const std::vector<int> & h_sub, const Eigen::MatrixXd & theta_hat, double RSS, int n_iterations): hSubset(h_sub){
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
// ************************   T R Y   -   A L L  -  P A I R S  *********************************************************
// *********************************************************************************************************************
/* Go through all pairs between J and M and calculate deltaRSS, save the smallest delta
 * together with indexes of that pair
 * */
void goThroughAllPairs(double & delta, int & iSwap, int & jSwap, const Eigen::MatrixXd & dataJX,
        const Eigen::MatrixXd & dataMX,
        const Eigen::MatrixXd & residuals,
        const Eigen::MatrixXd & inversion,
        const std::vector<int> & indexesJ,
        const std::vector<int> & indexesM) {

    unsigned h = dataJX.rows();
    unsigned nMinusH = dataMX.rows();

    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < nMinusH; ++j) {

            // calculate delta RSS - first prepare parameters
            double eI = residuals(indexesJ[i], 0); // na indexu i J
            double eJ = residuals(indexesM[j], 0); // na indexu j M
            double hII =  dataJX(i, Eigen::all) * inversion * (dataJX(i, Eigen::all).transpose())  ; // todo bude asi zle
            double hIJ =  dataJX(i, Eigen::all) * inversion * (dataMX(j, Eigen::all).transpose())  ;
            double hJJ =  dataMX(j, Eigen::all) * inversion * (dataMX(j, Eigen::all).transpose())  ;

            // next - do the calculation
            double nom = (eJ * eJ * (1 - hII) ) - ( eI * eI * (1 + hJJ)) + 2*eI*eJ*hIJ;
            double deNom = (1 - hII)*(1 + hJJ) + hIJ * hIJ;
            double newDelta = nom / deNom;

            if(newDelta < delta){
                delta = newDelta;
                iSwap = i;
                jSwap = j;
            }
        }
    }
}

// *********************************************************************************************************************
// ************************   R E F I N E M E N T   -    P R O C E S S   ***********************************************
// *********************************************************************************************************************
ResultFeasible * refinementProcess(std::vector<int> & indexesJ, std::vector<int> & indexesM, const Eigen::MatrixXd & X, const Eigen::MatrixXd & y ) {
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all); // okopiruji hodne dat -jinak ale indexuji 30x ve while
    int steps = 0;
    while(true) {
        // inversion =  (xT X).I ; theta = inversion * x.T * y ; esiduals = all y - all * theta
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_decomp(dataJX.transpose() * dataJX);
        Eigen::MatrixXd inversion = qr_decomp.inverse();
        Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);
        Eigen::MatrixXd theta = inversion * dataJX.transpose() * dataJy  ; // pxp * pxh  * hx1
        Eigen::MatrixXd residuals = y - X * theta;

        // go through all pairs
        double delta = 0;
        int iSwap;
        int jSwap;
        goThroughAllPairs(delta, iSwap, jSwap, dataJX, dataMX, residuals, inversion, indexesJ, indexesM);

        if(delta < 0){
           Eigen::MatrixXd tmp = dataJX(iSwap, Eigen::all); // swap data
           dataJX.row(iSwap).swap( dataMX.row(jSwap) );
           dataMX.row(jSwap).swap(tmp.row(0));
           int tmp_idx = indexesJ[iSwap]; // swap indexes also
           indexesJ[iSwap] = indexesM[jSwap];
           indexesM[jSwap] = tmp_idx;
           steps += 1; // inc number of steps
           if(steps > 1000) {
           std::cout << "while loop too long" << std::endl;
                break;
            }
           continue;
       }
       break;
    }
    // save the result - todo optimize this
    Eigen::MatrixXd yy = y(indexesJ, Eigen::all);
    Eigen::MatrixXd XX = X(indexesJ, Eigen::all);
    Eigen::HouseholderQR<Eigen::MatrixXd> finalDecomp(XX);
    Eigen::MatrixXd theta_final = finalDecomp.solve(yy);
    double rss =  ( (yy - XX * theta_final).transpose() * (yy - XX * theta_final) )(0,0);
    return new ResultFeasible(indexesJ, theta_final, rss, steps);
}

// *********************************************************************************************************************
// *********************   F E A S I B L E   -  S O L U T I O N   ******************************************************
// *********************************************************************************************************************
ResultFeasible* fs_lts(Eigen::MatrixXd X, Eigen::MatrixXd y, int numStarts, int hSize) {
    std::vector<ResultFeasible*> subsetResults;
    clock_t t;
    t = clock();

    unsigned N = X.rows();

    // for all stars
    for (int i = 0; i < numStarts; ++i) {
        // select initial random array of indexes |J| = p, |M|= n-p
        std::vector<int> permutation(N);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::random_shuffle ( permutation.begin(), permutation.end());
        std::vector<int> indexesJ(permutation.begin(), permutation.begin() + hSize);
        std::vector<int> indexesM(permutation.begin() + hSize, permutation.end() );  // todo - toto funguje ok

        // do the refinement process on (indexes J, indexes M , X, Y)
        ResultFeasible * result = refinementProcess(indexesJ, indexesM, X, y);

        // append result to result array
        subsetResults.push_back(result);
    }


    float time1 = ((float)(clock() - t))/CLOCKS_PER_SEC;

    // find and return best results
    ResultFeasible* best = subsetResults[0];
    for (int i = 0 ; i < numStarts; ++i)
        best = subsetResults[i]->rss < best->rss ? subsetResults[i] : best;
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

PYBIND11_MODULE(feasible_solution, m) {

	// documentation - not necessary
	 m.doc() = R"pbdoc(
		    Pybind11 feasible solution lts algorithm
		    -----------------------

		    .. currentmodule:: feasible_solution

		    .. autosummary::
		       :toctree: _generate

		       feasible_solution
		)pbdoc";


    // bind functions - and use documentation
    m.def("fs_lts", &fs_lts,  R"pbdoc(
        Feasible Solution Algorithm (FSA)
    )pbdoc");

    // bind classes -- warning !!! -- same class name (only in c++) leads to "generic type ClassName already registered"
    py::class_<ResultFeasible>(m, "ResultFeasible")
    .def(py::init< const std::vector<int> & , const Eigen::MatrixXd & , double ,int  >())
    .def("get_rss", &ResultFeasible::getRSS)
    .def("get_theta", &ResultFeasible::getTheta)
    .def("get_h_subset", &ResultFeasible::getHSubset)
    .def("get_n_inter", &ResultFeasible::getNIter)
    .def("get_time_1", &ResultFeasible::getTime1, "total time")
    .def("get_time_2", &ResultFeasible::getTime2, "unused")
    .def("get_time_3", &ResultFeasible::getTime3, "unused")
    ;

// version info - not neccesary
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
