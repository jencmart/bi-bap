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
#include <tuple>
#include <limits>

/*
system_clock - real time clock which is actually the clock used by system
high_resolution_clock - clock with the smallest tick/interval available/supported by the system
stable_clock - a clock with a steady tick rate (Recommended as tick rate is stable, while system_clock's tick period varies according to system load, see this
*/

namespace py = pybind11;


// result class for storing results that has bind to the python module
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
// *********************************************************************************************************************
// *********************************************************************************************************************
// ************************    I N V E R S I O N   V E R S I O N S   ***************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************


// *********************************************************************************************************************
// *************** S U B R O U T I N E S - INVERSION - FSA / MOEA / MMEA ***********************************************
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

// subroutine to update theta and inversion if row is excluded from X (Agullo)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> theta_and_inversion_minus(const Eigen::MatrixXd & theta_plus,
                                            const Eigen::MatrixXd & dataJX, const Eigen::MatrixXd & dataJy,
                                            int iSwap, const Eigen::MatrixXd & inversion_plus){
    double jmj =  idx_inv_idx(dataJX, iSwap, inversion_plus);
    Eigen::MatrixXd xj = dataJX(iSwap, Eigen::all);
    Eigen::MatrixXd yj = dataJy(iSwap, Eigen::all);

    double w = -1 / (1 + jmj);
    Eigen::MatrixXd u = inversion_plus * xj.transpose();
    Eigen::MatrixXd theta_minus = theta_plus + ((yj - (xj*theta_plus)  )(0, 0) * (w*u));
    Eigen::MatrixXd inversion_minus = inversion_plus - (w * u * u.transpose());   // O(p^2)

    return std::make_tuple(theta_minus, inversion_minus);
}

// subroutine to calculate residual sum of squares
double calculateRSS(const Eigen::MatrixXd & dataJX, const Eigen::MatrixXd & dataJy, const Eigen::MatrixXd & theta ){
    Eigen::MatrixXd residuals = dataJy - dataJX * theta;
    double rss = (residuals.transpose() * residuals )(0,0);
    return rss;
}

// subroutine performing swap row xi from matrix J with row xj from matrix M
void swap_observations(Eigen::MatrixXd & dataJX, Eigen::MatrixXd & dataJy,
                        Eigen::MatrixXd & dataMX, Eigen::MatrixXd & dataMy,
                        int iSwap, int jSwap, std::vector<int> & indexesJ, std::vector<int> & indexesM){
    Eigen::MatrixXd tmp = dataJX(iSwap, Eigen::all); // swap data X
    dataJX.row(iSwap).swap(dataMX.row(jSwap));
    dataMX.row(jSwap).swap(tmp.row(0));

    Eigen::MatrixXd tmp2 = dataJy(iSwap, Eigen::all); // swap data y
    dataJy.row(iSwap).swap(dataMy.row(jSwap));
    dataMy.row(jSwap).swap(tmp2.row(0));

    int tmp_idx = indexesJ[iSwap]; // swap indexes also
    indexesJ[iSwap] = indexesM[jSwap];
    indexesM[jSwap] = tmp_idx;
}

// subroutine finding observation from matrix M, which if included to matrix J will increase RSS the smallest (Agullo)
std::tuple<int, double> smallest_include_inv(const Eigen::MatrixXd & dataMX, const Eigen::MatrixXd & dataMy,
                                             const Eigen::MatrixXd & theta, const Eigen::MatrixXd & inversion){
    double gamma_plus_min = std::numeric_limits<double>::infinity();
    int jSwap = 0;

    unsigned n = dataMX.rows();

    for(unsigned j = 0 ; j < n ; ++j ){

        // calculate gamma+ O(p^2)
        double imi = idx_inv_idx(dataMX, j, inversion);
        Eigen::MatrixXd xi = dataMX(j, Eigen::all);
        Eigen::MatrixXd yi = dataMy(j, Eigen::all);
        double gamma_plus = std::pow((yi - xi*theta)(0, 0), 2) / (1 + imi);
        // update if smaller
        if(gamma_plus < gamma_plus_min){
            gamma_plus_min = gamma_plus;
            jSwap = j;
        }
    }

    return std::make_tuple(jSwap, gamma_plus_min);
}

// subroutine finding observation from matrix J, which if excluded, will reduce RSS the most
std::tuple<int, double> greatest_exclude_inv(const Eigen::MatrixXd & dataJX, const Eigen::MatrixXd & dataJy,
                                             const Eigen::MatrixXd & theta, const Eigen::MatrixXd & inversion){
    double gamma_minus_max = -1;
    int iSwap = 0;
    unsigned n = dataJX.rows();
    for(unsigned i = 0 ; i < n ; ++i ){

        // calculate gamma- O(p^2)
        double jmj = idx_inv_idx(dataJX, i, inversion);
        Eigen::MatrixXd xj = dataJX(i, Eigen::all);
        Eigen::MatrixXd yj = dataJy(i, Eigen::all);
        double gamma_minus = std::pow((yj - xj*theta)(0, 0), 2) / (1 - jmj);
        // update if greater
        if(gamma_minus > gamma_minus_max){
            gamma_minus_max = gamma_minus;
            iSwap = i;
        }
    }

    return std::make_tuple(iSwap, gamma_minus_max);
}


// *********************************************************************************************************************
// ************************  F S A - I N V -   T R Y   -   A L L  -  P A I R S  ****************************************
// *********************************************************************************************************************
/* Go through all pairs between J and M and calculate deltaRSS, save the smallest delta
 * together with indexes of that pair
 * */
void goThroughAllPairsFsaInv(double & delta, int & iSwap, int & jSwap, const Eigen::MatrixXd & dataJX,
        const Eigen::MatrixXd & dataMX,
        const Eigen::MatrixXd & residuals,
        const Eigen::MatrixXd & inversion,
        const std::vector<int> & indexesJ,
        const std::vector<int> & indexesM) {

    unsigned h = dataJX.rows();
    unsigned nMinusH = dataMX.rows();
    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < nMinusH; ++j) {

            // calculate delta RSS (Hawkins)

            // prepare params
            double eI = residuals(indexesJ[i], 0); //  residual for excluded row
            double eJ = residuals(indexesM[j], 0); //  residual for included row
            double hII =  dataJX(i, Eigen::all) * inversion * (dataJX(i, Eigen::all).transpose());
            double hIJ =  dataJX(i, Eigen::all) * inversion * (dataMX(j, Eigen::all).transpose());
            double hJJ =  dataMX(j, Eigen::all) * inversion * (dataMX(j, Eigen::all).transpose());

            // perform calculation
            double nom = (eJ * eJ * (1 - hII) ) - ( eI * eI * (1 + hJJ)) + 2*eI*eJ*hIJ;
            double deNom = (1 - hII)*(1 + hJJ) + hIJ * hIJ;
            double newDelta = nom / deNom;

            // update indexes and value if smaller
            if(newDelta < delta){
                delta = newDelta;
                iSwap = i;
                jSwap = j;
            }
        }
    }
}
// *********************************************************************************************************************
// ************************ F S A - I N V -  R E F I N E M E N T   -    P R O C E S S   ********************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessFsaInv(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // Create the sub-matrices
    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    Eigen::MatrixXd theta, inversion;

    int steps = 0;
    for(int it = 0 ; it < maxSteps ; it++) {

        // calculate theta and inversion  # O(np^2)
        std::tie(theta, inversion) = theta_and_inversion(dataJX, dataJy);

        // calculate residuals r_1 ... r_n    O(np)
        Eigen::MatrixXd residuals = y - X * theta;

        // find the optimal swap - j add ; i remove O(n^2p^2)
        double delta = 0;
        int iSwap, jSwap;
        goThroughAllPairsFsaInv(delta, iSwap, jSwap, dataJX, dataMX, residuals, inversion, indexesJ, indexesM);

        // strong necessary condition satisfied
        if(!(delta < 0))
            break;

        // swap observations
        swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);

        // step ++
        steps += 1;
    }

    // calculate theta and inversion  # O(np^2)
    std::tie(theta, inversion) = theta_and_inversion(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);

    return new ResultFeasible(indexesJ, theta, rss, steps);
}




// *********************************************************************************************************************
// ************************  M O E A - I N V -   T R Y   -   A L L  -  P A I R S  **************************************
// *********************************************************************************************************************
/* Go through all pairs between J and M and calculate deltaRSS, save the smallest delta
 * together with indexes of that pair
 * */
 std::vector<double> all_idx_inv_idx(const Eigen::MatrixXd & dataMatrix, const Eigen::MatrixXd & inversion){
    unsigned n = dataMatrix.rows();

    std::vector<double> arr_idx_inv_idx;
    for(unsigned i = 0 ; i < n ; ++i ){
        double xmx =  idx_inv_idx(dataMatrix, i, inversion);
        arr_idx_inv_idx.push_back(xmx);
    }

    return arr_idx_inv_idx;
}

void goThroughAllPairsMoeaInv(double & rho, int & iSwap, int & jSwap, const Eigen::MatrixXd & dataJX,
        const Eigen::MatrixXd & dataMX,
        const Eigen::MatrixXd & residuals,
        const Eigen::MatrixXd & inversion,
        const std::vector<int> & indexesJ,
        const std::vector<int> & indexesM,
        double rss) {

    // (moea speedup)
    double ro_b_min = 1;

    // calculate imi and jmj in advance O(p^2n)
    std::vector<double> arr_imi = all_idx_inv_idx(dataMX, inversion); // for included rows
    std::vector<double> arr_jmj = all_idx_inv_idx(dataJX, inversion); // for excluded rows

    unsigned h = dataJX.rows();
    unsigned nMinusH = dataMX.rows();
    for (unsigned i = 0; i < h; ++i) {
        double jmj = arr_jmj[i];
        for (unsigned j = 0; j < nMinusH; ++j) {
            double imi = arr_imi[j];

            // calculate delta RSS - first prepare parameters
            double ei = residuals(indexesM[j], 0);  // residual for included row
            double ej = residuals(indexesJ[i], 0);  // residual for excluded row

            // calculate ro_b (moea speedup)
            double a = ((1 + imi + ( std::pow(ei,2) / rss)) * (1 - jmj - (std::pow(ej,2) / rss)));
            double b = (1 + imi - jmj);
            double ro_b = a / b;

            if(ro_b > ro_b_min)
                continue;

            // calculate ro_i_j multiplicative difference (Agullo)
            double i_m_j = (dataMX(j, Eigen::all) * inversion * (dataJX(i, Eigen::all).transpose()))(0, 0);

            a = a + (std::pow((i_m_j + (ei * ej) / rss), 2));
            b = b + (std::pow(i_m_j, 2)) - imi * jmj;
            double newRo = a / b;

            if(newRo < ro_b_min)
                ro_b_min = newRo;

            if(newRo < rho){
                rho = newRo;
                iSwap = i;
                jSwap = j;
            }
        }
    }
}
// *********************************************************************************************************************
// ************************ M O E A - I N V -  R E F I N E M E N T   -    P R O C E S S   ******************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessMoeaInv(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {

    int steps = 0;
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // calculate theta and inversion  # O(np^2)
    Eigen::MatrixXd theta, inversion;
    std::tie(theta, inversion) = theta_and_inversion(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);

    // shortcut
    if(indexesM.empty()){
        return new ResultFeasible(indexesJ, theta, rss, steps);
    }

    // Create the sub-matrices
    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    for(int it = 0 ; it < maxSteps ; it++) {

        // calculate residuals r_1 ... r_n    O(np)
        Eigen::MatrixXd residuals = y - X * theta;

        // find the optimal swap - j add ; i remove O(n^2p^2)
        double rho = 1;
        int iSwap, jSwap;
        goThroughAllPairsMoeaInv(rho, iSwap, jSwap, dataJX, dataMX, residuals, inversion, indexesJ, indexesM, rss);

        // strong necessary condition satisfied
        if(rho >= 1){
            break;
        }else{
            // update rss
            rss = rss*rho;
            // update theta -> theta_plus ; inversion -> inversion_plus  O(p^2)
            Eigen::MatrixXd theta_plus, inversion_plus;
            std::tie(theta_plus, inversion_plus) = theta_and_inversion_plus(theta, dataMX, dataMy, jSwap, inversion);
            // update theta_plus -> theta_minus ; inversion_plus -> inversion_minus  O(p^2)
            std::tie(theta, inversion) = theta_and_inversion_minus(theta_plus, dataJX, dataJy, iSwap, inversion_plus);
            // swap observations
            swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);
            // step ++
            steps += 1;
        }
   }
    return new ResultFeasible(indexesJ, theta, rss, steps);
}



// *********************************************************************************************************************
// ************************ M M E A - I N V -  R E F I N E M E N T   -    P R O C E S S   ******************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessMmeaInv(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {

    int steps = 0;
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // calculate theta and inversion  # O(np^2)
    Eigen::MatrixXd theta, inversion;
    std::tie(theta, inversion) = theta_and_inversion(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);

    // shortcut
    if(indexesM.empty()){
        return new ResultFeasible(indexesJ, theta, rss, steps);
    }

    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    for(int it = 0 ; it < maxSteps ; it++) {

        // find optimal include  O(p^2n)
        int jSwap;
        double gamma_plus;
        std::tie(jSwap, gamma_plus) = smallest_include_inv(dataMX, dataMy, theta, inversion);


        // update theta -> theta_plus ; inversion -> inversion_plus  O(p^2)
        Eigen::MatrixXd theta_plus, inversion_plus;
        std::tie(theta_plus, inversion_plus) = theta_and_inversion_plus(theta, dataMX, dataMy, jSwap, inversion);

        // find the optimal exclude (no need to update J ... worst case: gamma_plus == gamma_minus )  O(p^2n)
        int iSwap;
        double gamma_minus;
        std::tie(iSwap, gamma_minus) = greatest_exclude_inv(dataJX, dataJy, theta_plus, inversion_plus);

        // improvement cannot be made
        if(!(gamma_plus < gamma_minus))
            break;

        // update theta, inversion, rss, J, M, residualsJ residualsM

        // update rss
        rss = rss + gamma_plus - gamma_minus;

        // update theta_plus -> theta_minus ; inversion_plus -> inversion_minus  O(p^2)
        std::tie(theta, inversion) = theta_and_inversion_minus(theta_plus, dataJX, dataJy, iSwap, inversion_plus);


        // swap observations
        swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);
        // step ++
        steps += 1;
   }
    return new ResultFeasible(indexesJ, theta, rss, steps);
}








// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// ************************    Q R   V E R S I O N S   *****************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************


// *********************************************************************************************************************
// *************** S U B R O U T I N E S - QR - FSA / MOEA / MMEA ******************************************************
// *********************************************************************************************************************

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> theta_qr(const Eigen::MatrixXd & dataJX, const Eigen::MatrixXd & dataJy){
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(dataJX);
    Eigen::MatrixXd q = qr.householderQ();
    Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>(); // represents q.transpose()*dataJX
    Eigen::MatrixXd r1 = r.topRows(r.cols());

    Eigen::MatrixXd theta = qr.solve(dataJy);

    return std::make_tuple(theta, q, r, r1);
}

// subroutine to solving triangular system R.T v = x for v and consequently calculating dot product v v.T
std::tuple<double, Eigen::MatrixXd> idx_qr_idx(const Eigen::MatrixXd & dataMatrix, int idx, const Eigen::MatrixXd & r1){

    Eigen::MatrixXd x = dataMatrix(idx, Eigen::all);

    Eigen::MatrixXd r1trans = r1.transpose();
    // transpose upper triangular and solve lower triangular system
    Eigen::MatrixXd vi = r1trans.triangularView<Eigen::Lower>().solve(x.transpose());

    double xmx = (vi.transpose() * vi)(0, 0); // vi.T * vi
   //Eigen::MatrixXd xmx2 = (vi.transpose() * vi); // vi.T * vi

//    if(xmx > 1) {
//    std::cout << r1 << std::endl;
//        std::cout << std::endl;
//        std::cout << r1trans << std::endl;
//        std::cout << std::endl;
//
//        std::cout << vi << std::endl;
//        std::cout << std::endl;
//        std::cout << x << std::endl;
//        std::cout << std::endl;
//        exit(12);
//    }




    return std::make_tuple(xmx, vi);
}

// subroutine for solving triangular system r1 u = vi and product x u.T  for two different observations
double idx_qr_j(const Eigen::MatrixXd & dataJX, int i, const Eigen::MatrixXd & r1, const Eigen::MatrixXd & vi){

    Eigen::MatrixXd x = dataJX(i, Eigen::all);
    Eigen::MatrixXd u = r1.triangularView<Eigen::Upper>().solve(vi);
    double imj = (x * u)(0,0); // x * u
    return imj;
}

// subroutine finding observation from matrix M, which if included to matrix J will increase RSS the smallest (Agullo)
std::tuple<int, double> smallest_include_qr(const Eigen::MatrixXd & dataMX, const Eigen::MatrixXd & dataMy,
                                             const Eigen::MatrixXd & theta, const Eigen::MatrixXd & r1){
    double gamma_plus_min = std::numeric_limits<double>::infinity();
    int jSwap = 0;

    unsigned n = dataMX.rows();

    for(unsigned j = 0 ; j < n ; ++j ){

        // calculate gamma+ O(p^2)
        double imi;
        Eigen::MatrixXd vi; // not needed
        std::tie(imi, vi) =  idx_qr_idx(dataMX, j, r1);

        Eigen::MatrixXd xi = dataMX(j, Eigen::all);
        Eigen::MatrixXd yi = dataMy(j, Eigen::all);
        double gamma_plus = std::pow((yi - xi*theta)(0, 0), 2) / (1 + imi);


        // update if smaller
        if(gamma_plus < gamma_plus_min){
            gamma_plus_min = gamma_plus;
            jSwap = j;
        }
    }
    return std::make_tuple(jSwap, gamma_plus_min);
}


// subroutine finding observation from matrix J, which if excluded, will reduce RSS the most
std::tuple<int, double> greatest_exclude_qr(const Eigen::MatrixXd & dataJX, const Eigen::MatrixXd & dataJy,
                                             const Eigen::MatrixXd & theta, const Eigen::MatrixXd & r1){
    double gamma_minus_max = -1;
    int iSwap = 0;
    unsigned n = dataJX.rows();
    for(unsigned i = 0 ; i < n ; ++i ){

        // calculate gamma- O(p^2)
        double jmj;
        Eigen::MatrixXd vj; // not needed
        std::tie(jmj, vj) =  idx_qr_idx(dataJX, i, r1);

        Eigen::MatrixXd xj = dataJX(i, Eigen::all);
        Eigen::MatrixXd yj = dataJy(i, Eigen::all);
        double gamma_minus = std::pow((yj - xj*theta)(0, 0), 2) / (1 - jmj);
        // update if greater
        if(gamma_minus > gamma_minus_max){
            gamma_minus_max = gamma_minus;
            iSwap = i;
        }
    }

    return std::make_tuple(iSwap, gamma_minus_max);
}




// subroutine for calculating sinus(fi) and cosinus(fi) between two points
void calculate_cos_sin(double f, double g, double & cos, double & sin, double & R){
    if(g == 0){
        cos = 1;
        sin = 0;
        R = f;
    } else if(f == 0){
        cos = 0;
        sin = 1;
        R = g;
    }else{
        R = std::sqrt(std::pow(f, 2) + std::pow(g, 2));
        cos = f / R;
        sin = g / R;
    }

    if((std::fabs(f) > std::fabs(g)) && cos < 0){
        cos = -cos;
        sin = -sin;
        R = -R;
    }
}

// update QR decomposition when row is inserted using the Givens rotations
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> qr_insert(const Eigen::MatrixXd & q, const Eigen::MatrixXd & r, const Eigen::MatrixXd & row, int idx){

    // create and fill new matrix Q
    Eigen::MatrixXd qnew = Eigen::MatrixXd::Zero(q.rows()+1, q.cols()+1);
    qnew.block(0,0,q.rows(),q.rows()) = q.block(0,0,q.rows(),q.rows());
    qnew(qnew.rows()-1, qnew.cols()-1) = 1;

    // create and fill new matrix R
    Eigen::MatrixXd rnew = Eigen::MatrixXd::Zero(r.rows()+1, r.cols());
    rnew.topRows(r.rows()) = r.topRows(r.rows());
    rnew.bottomRows(1) =  row.topRows(1);

    // set the limit
    int n = qnew.rows();
    int p = r.cols();
    int limit = ((n-1 < p) ? n-1 : p); // although we assume n >= p


    // create additional Givens matrices..
    for(int j = 0; j < limit; ++j){ // 0, 1, 2 .... p-1 OK

        // we create zeroes on the last row against values on the diagonal of R
        double cos, sin, R;
        calculate_cos_sin(rnew(j, j), rnew(n-1, j), cos, sin, R);  // edge of triangle , last row
        rnew(j, j) = R;  // rotated value
        rnew(n-1, j) = 0;  // numerical stability...

        // rotate rnew ... multiply with givens matrix (G := givens(j, n-1)  ;  R = G R )
        //Eigen::MatrixXd rowX = rnew(j, Eigen::all);   // row we create zeroes against
        //Eigen::MatrixXd rowY = rnew(n-1, Eigen::all); // row to be zeroed
        for(int i = j+1; i < p ; ++i){ // we already set r(j,j) and r(n-1,j) above...
            double temp =    cos * rnew(j, i) + sin * rnew(n-1, i);
            rnew(n - 1, i) = cos * rnew(n-1, i) - sin * rnew(j, i);  // Y
            rnew(j, i) = temp;  // X
        }


        // multiply matrix Q ...  (G := givens(j, n-1)  ;  Q = Q G.T )
        //Eigen::MatrixXd q_colX = qnew(Eigen::all, j);    // jth column (length n)
        //Eigen::MatrixXd q_colY = qnew(Eigen::all, n-1);  // last column (length n)

        for(int i = 0; i < n; ++i){  // i in range(n):  # whole cols...
            double temp =  cos * qnew(i, j) + sin * qnew(i, n-1);
            qnew(i, n-1) = cos * qnew(i, n-1) - sin * qnew(i, j); // Y
            qnew(i, j) = temp;  // X
        }
    }

    // move the last (inserted) row to the correct position (row idx (that means at index idx-1)
    // put it behind the row we consequently remove ...
    for(int j = n-1; j > idx; j-- ){ // n-1, n-2 ... idx+1
        qnew.row(j).swap(qnew.row(j-1)); // propagate last row up
    }

    rnew = rnew.triangularView<Eigen::Upper>();
    return std::make_tuple(qnew, rnew);
}



// update QR decomposition when row is excluded using the Givens rotations
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> qr_delete(const Eigen::MatrixXd & qa, const Eigen::MatrixXd & ra, int idx){

    Eigen::MatrixXd qnew = qa;
    Eigen::MatrixXd rnew = ra;

    // num rows to delete
    int p_del = 1;

    // propagate the row we remove to the first position
    if(idx != 0){
        for(int j = idx; j > 0; --j ){ // (idx, ... 2, 1)
            qnew.row(j).swap(qnew.row(j-1)); // propagate row up
        }
    }

    int n = qnew.rows();
    int p = rnew.cols();


    // we want to introduce zeroes in the first row od Q at indexes 1, 2 .... n-1
    for(int j = n-2; j > -1 ; --j){   // n-2 ... 1, 0

        // one by one from the end ... (we can imagine first rows as column vector)
        // so first we zero j+1 , in next iteration j etc...
        double cos, sin, R;
        calculate_cos_sin(qnew(0, j), qnew(0, j+1), cos, sin, R);
        qnew(0, j) = R;
        qnew(0, j+1) = 0;

        // multiply R if non-zero row .. O(p)
        if(j < p){
            //Eigen::MatrixXd rowX = rnew(j, Eigen::all);  // row
            //Eigen::MatrixXd rowY = rnew(j+1, Eigen::all); // row
            for(int i = j; i < p; i++){ // rotate only non-zero part of the row...
                double temp =    cos * rnew(j, i) + sin * rnew(j+1, i);
                rnew(j+1,i) =    cos * rnew(j+1, i) - sin * rnew(j, i);  // Y
                rnew(j, i) = temp; // X
            }
        }

        // multiply Q
        //Eigen::MatrixXd q_colX = qnew(Eigen::all, j);    // jth column (length n)
        //Eigen::MatrixXd q_colY = qnew(Eigen::all, j+1);  // (j+1)th  column (length n)
        for(int i = p_del; i < n; i++ ){  // rows 1, 2, ... , n-1
            double temp =  cos * qnew(i, j) + sin * qnew(i, j+1);
            qnew(i, j+1) = cos * qnew(i, j+1) - sin * qnew(i, j); // Y
            qnew(i, j) = temp;  // X
        }
    }

    qnew = qnew.block(1, 1, n-1, n-1);
    rnew = rnew.bottomRows(n-1);
    rnew = rnew.triangularView<Eigen::Upper>();
    return std::make_tuple(qnew, rnew);
}




// *********************************************************************************************************************
// ************************  F S A - Q R -   T R Y   -   A L L  -  P A I R S  ****************************************
// *********************************************************************************************************************
/* Go through all pairs between J and M and calculate deltaRSS, save the smallest delta
 * together with indexes of that pair
 * */
void goThroughAllPairsFsaQr(double & delta, int & iSwap, int & jSwap, const Eigen::MatrixXd & dataJX,
        const Eigen::MatrixXd & dataMX,
        const Eigen::MatrixXd & residuals,
        const Eigen::MatrixXd & r1,
        const std::vector<int> & indexesJ,
        const std::vector<int> & indexesM) {

    unsigned h = dataJX.rows();
    unsigned nMinusH = dataMX.rows();
    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < nMinusH; ++j) {

            // calculate delta RSS (Hawkins)

            // prepare params
            double eI = residuals(indexesJ[i], 0); //  residual for excluded row
            double eJ = residuals(indexesM[j], 0); //  residual for included row

            double imi, jmj, imj;
            Eigen::MatrixXd vi, vj;

            std::tie(imi, vi) =  idx_qr_idx(dataMX, j, r1);
            std::tie(jmj, vj) =  idx_qr_idx(dataJX, i, r1);
            imj = idx_qr_j(dataJX, i, r1, vi);

            double hII =  jmj;
            double hIJ =  imi;
            double hJJ =  imj;

            // perform calculation
            double nom = (eJ * eJ * (1 - hII) ) - ( eI * eI * (1 + hJJ)) + 2*eI*eJ*hIJ;
            double deNom = (1 - hII)*(1 + hJJ) + hIJ * hIJ;
            double newDelta = nom / deNom;

            // update indexes and value if smaller
            if(newDelta < delta){
                delta = newDelta;
                iSwap = i;
                jSwap = j;
            }
        }
    }
}

// *********************************************************************************************************************
// ************************ F S A - Q R -  R E F I N E M E N T   -    P R O C E S S   **********************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessFsaQr(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // Create the sub-matrices
    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    Eigen::MatrixXd theta, q, r, r1;

    int steps = 0;
    for(int it = 0 ; it < maxSteps ; it++) {

        // calculate theta and QR decomposition  O(p^2n)
        std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);

        // calculate residuals r_1 ... r_n    O(np)
        Eigen::MatrixXd residuals = y - X * theta;

        // find the optimal swap - j add ; i remove O(n^2p^2)
        double delta = 0;
        int iSwap, jSwap;
        goThroughAllPairsFsaQr(delta, iSwap, jSwap, dataJX, dataMX, residuals, r1, indexesJ, indexesM);

        // strong necessary condition satisfied
        if(!(delta < 0))
            break;

        // swap observations
        swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);


        // step ++
        steps += 1;
    }

    // calculate theta and QR decomposition  O(p^2n)
    std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);

    return new ResultFeasible(indexesJ, theta, rss, steps);
}




// *********************************************************************************************************************
// ************************  M O E A - Q R -   T R Y   -   A L L  -  P A I R S  ****************************************
// *********************************************************************************************************************
/* Go through all pairs between J and M and calculate deltaRSS, save the smallest delta
 * together with indexes of that pair
 * */
 std::tuple<std::vector<double>, std::vector<Eigen::MatrixXd>> all_idx_idx_qr(const Eigen::MatrixXd & dataMatrix, const Eigen::MatrixXd & r1){

    std::vector<double> arr_idx_qr_idx;
    std::vector<Eigen::MatrixXd> arr_vi;

    unsigned n = dataMatrix.rows();
    for(unsigned i = 0 ; i < n ; ++i ){

        double xmx;
        Eigen::MatrixXd vi;
        std::tie(xmx, vi) =  idx_qr_idx(dataMatrix, i, r1);
        arr_idx_qr_idx.push_back(xmx);
        arr_vi.push_back(vi);
    }

    return std::make_tuple(arr_idx_qr_idx, arr_vi);
}

void goThroughAllPairsMoeaQr(double & rho, int & iSwap, int & jSwap, const Eigen::MatrixXd & dataJX,
        const Eigen::MatrixXd & dataMX,
        const Eigen::MatrixXd & residuals,
        const Eigen::MatrixXd & r1,
        const std::vector<int> & indexesJ,
        const std::vector<int> & indexesM,
        double rss) {

    // (moea speedup)
    double ro_b_min = 1;

    // calculate imi and jmj in advance O(p^2n)

    std::vector<double> arr_imi;
    std::vector<Eigen::MatrixXd> arr_vi;
    std::tie(arr_imi, arr_vi) = all_idx_idx_qr(dataMX, r1); // for included rows

    std::vector<double> arr_jmj;
    std::vector<Eigen::MatrixXd> arr_vj;
    std::tie(arr_jmj, arr_vj) = all_idx_idx_qr(dataJX, r1); // for excluded rows

    unsigned h = dataJX.rows();
    unsigned nMinusH = dataMX.rows();
    for (unsigned i = 0; i < h; ++i) {
        double jmj = arr_jmj[i];
        for (unsigned j = 0; j < nMinusH; ++j) {
            double imi = arr_imi[j];
            Eigen::MatrixXd vi = arr_vi[j];

            // calculate delta RSS - first prepare parameters
            double ei = residuals(indexesM[j], 0);  // residual for included row
            double ej = residuals(indexesJ[i], 0);  // residual for excluded row

            // calculate ro_b (moea speedup)
            double a = ((1 + imi + ( std::pow(ei,2) / rss)) * (1 - jmj - (std::pow(ej,2) / rss)));
            double b = (1 + imi - jmj);
            double ro_b = a / b;

            if(ro_b < 0) {
                std::cout << "-----" << std::endl;
                std::cout << ro_b << std::endl;
                std::cout << a << std::endl;
                std::cout << b << std::endl;
                std::cout << std::endl;
                std::cout << imi << std::endl;
                std::cout << jmj << std::endl;
                exit(21);
            }

            if(ro_b > ro_b_min)
                continue;

            // calculate ro_i_j multiplicative difference (Agullo)
            double i_m_j = idx_qr_j(dataJX, i, r1, vi);

            a = a + (std::pow((i_m_j + (ei * ej) / rss), 2));
            b = b + (std::pow(i_m_j, 2)) - imi * jmj;
            double newRo = a / b;


          if(newRo < 0) {
                std::cout << "-----" << std::endl;
                std::cout << newRo << std::endl;
                std::cout << a << std::endl;
                std::cout << b << std::endl;
                std::cout << std::endl;
                std::cout << i_m_j << std::endl;

                exit(22);
            }


            if(newRo < ro_b_min)
                ro_b_min = newRo;

            if(newRo < rho){
                rho = newRo;
                iSwap = i;
                jSwap = j;
            }
        }
    }
}

// *********************************************************************************************************************
// ************************ M O E A - Q R -  R E F I N E M E N T   -    P R O C E S S   ********************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessMoeaQr(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {

    int steps = 0;
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // calculate theta and QR decomposition  O(p^2n)
    Eigen::MatrixXd theta, q, r, r1;
    std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);
    //double rss1 = rss;

    // shortcut
    if(indexesM.empty()){
        return new ResultFeasible(indexesJ, theta, rss, steps);
    }

    // Create the sub-matrices
    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    for(int it = 0 ; it < maxSteps ; it++) {

        // calculate residuals r_1 ... r_n    O(np)
        Eigen::MatrixXd residuals = y - X * theta;

        // find the optimal swap - j add ; i remove O(n^2p^2)
        double rho = 1;
        int iSwap, jSwap;
        goThroughAllPairsMoeaQr(rho, iSwap, jSwap, dataJX, dataMX, residuals, r1, indexesJ, indexesM, rss);

        // strong necessary condition satisfied
        if(rho >= 1){
            break;
        }else{
            // update rss
            rss = rss*rho;

            // row to insert
            // Eigen::MatrixXd rowToInsert = dataMX(jSwap, Eigen::all);

            // update QR O(np^2) and QR O(n^2)
            // Eigen::MatrixXd q_plus, r_plus;
            // std::tie(q_plus, r_plus) = qr_insert(q, r, rowToInsert, iSwap+1);

            // CORRECT DECOMPOSITION ON DATA PLUS
            // Eigen::MatrixXd dataJX_plus = Eigen::MatrixXd::Zero(dataJX.rows()+1, r.cols());
            // dataJX_plus.topRows(iSwap+1) = dataJX.topRows(iSwap+1); // ok
            // dataJX_plus.row(iSwap+1) = rowToInsert.topRows(1);
            // dataJX_plus.bottomRows(dataJX.rows() - iSwap-1) = dataJX.bottomRows(dataJX.rows() - iSwap-1);

            // Eigen::MatrixXd q_minus, r_minus;
            // std::tie(q_minus, r_minus) = qr_delete(q_plus, r_plus, iSwap);

            // swap observations
            swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);

            std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);
            std::cout << rss << std::endl;
            // step ++
            steps += 1;
        }
   }
    return new ResultFeasible(indexesJ, theta, rss, steps);
}


// *********************************************************************************************************************
// ************************ M M E A - Q R -  R E F I N E M E N T   -    P R O C E S S   ********************************
// *********************************************************************************************************************
ResultFeasible * refinementProcessMmeaQr(std::vector<int> & indexesJ, std::vector<int> & indexesM,
                                        const Eigen::MatrixXd & X, const Eigen::MatrixXd & y, int maxSteps ) {

    int steps = 0;
    // Create the sub-matrices
    Eigen::MatrixXd dataJX = X(indexesJ, Eigen::all);
    Eigen::MatrixXd dataJy = y(indexesJ, Eigen::all);

    // calculate theta and QR decomposition  O(p^2n)
    Eigen::MatrixXd theta, q, r, r1;
    std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);

    // calculate RSS O(np)
    double rss = calculateRSS(dataJX, dataJy, theta);

    // shortcut
    if(indexesM.empty()){
        return new ResultFeasible(indexesJ, theta, rss, steps);
    }

    Eigen::MatrixXd dataMX = X(indexesM, Eigen::all);
    Eigen::MatrixXd dataMy = y(indexesM, Eigen::all);

    for(int it = 0 ; it < maxSteps ; it++) {

        // find optimal include  O(p^2n)
        int jSwap;
        double gamma_plus;
        std::tie(jSwap, gamma_plus) = smallest_include_qr(dataMX, dataMy, theta, r1);


        // create JX_plus
        Eigen::MatrixXd dataJX_plus(dataJX.rows()+1, dataJX.cols());
        dataJX_plus.topRows(dataJX.rows()) = dataJX.topRows(dataJX.rows());
        dataJX_plus.bottomRows(1) =  dataMX(jSwap, Eigen::all);

        // create JY_plus
        Eigen::MatrixXd dataJy_plus(dataJy.rows()+1, dataJy.cols());
        dataJy_plus.topRows(dataJy.rows()) = dataJy.topRows(dataJy.rows());
        dataJy_plus.bottomRows(1) =  dataMy(jSwap, Eigen::all);

        // update theta -> theta_plus ; qr -> qr_plus  O(p^2)
        Eigen::MatrixXd theta_plus, q_plus, r_plus, r1_plus;
        std::tie(theta_plus, q_plus, r_plus, r1_plus) = theta_qr(dataJX_plus, dataJy_plus);

        // find the optimal exclude (no need to update J ... worst case: gamma_plus == gamma_minus )  O(p^2n)
        int iSwap;
        double gamma_minus;
        std::tie(iSwap, gamma_minus) = greatest_exclude_qr(dataJX, dataJy, theta_plus, r1_plus);

        // improvement cannot be made
        if(!(gamma_plus < gamma_minus))
            break;

        // update theta, qr, rss, J, M, residualsJ residualsM

        // update rss
        rss = rss + gamma_plus - gamma_minus;

        // swap observations
        swap_observations(dataJX, dataJy, dataMX, dataMy, iSwap, jSwap, indexesJ, indexesM);

        // update theta q, r, r1
        std::tie(theta, q, r, r1) = theta_qr(dataJX, dataJy);

        // step ++
        steps += 1;
   }
    return new ResultFeasible(indexesJ, theta, rss, steps);
}

// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************   F E A S I B L E   -  S O L U T I O N  - M A I N *********************************************
// *********************************************************************************************************************
// *********************************************************************************************************************
// *********************************************************************************************************************

// Perform calculation of LTS estimate satisfying strong necessary condition (known as feasible solution)n
ResultFeasible* fs_lts(Eigen::MatrixXd X, Eigen::MatrixXd y, int numStarts, int maxSteps, int hSize, int alg, int calc) {
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
        std::vector<int> indexesM(permutation.begin() + hSize, permutation.end() );

        ResultFeasible * result;

        if( calc == 0){
            if(alg == 0){
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessFsaInv(indexesJ, indexesM, X, y, maxSteps);
            }else if(alg == 1){
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessMoeaInv(indexesJ, indexesM, X, y, maxSteps);
            }else {
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessMmeaInv(indexesJ, indexesM, X, y, maxSteps);
            }
        }else{
            if(alg == 0){
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessFsaQr(indexesJ, indexesM, X, y, maxSteps);
            }else if(alg == 1){
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessMoeaQr(indexesJ, indexesM, X, y, maxSteps);
            }else {
                // do the refinement process on (indexes J, indexes M , X, Y)
                result = refinementProcessMmeaQr(indexesJ, indexesM, X, y, maxSteps);
            }
        }

        // append result to result array
        subsetResults.push_back(result);
    }

    float time1 = ((float)(clock() - t))/CLOCKS_PER_SEC;

    // find and return best result
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

// used to bind code to the python module

// small info:
// # To work way you properly, you have to adhere the naming convention right:
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
        Algorithms calculating the LTS estimate satisfying strong necessary condition.
        Algorithm options: fsa, moea, mmea
        Calculation options: inversion, qr decomposition
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
