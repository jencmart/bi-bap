/*
<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['/home/jencmart/bi-bap/implementace/scprits/python-pybind11/eigen-git-mirror']

setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/LU>

namespace py = pybind11;


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

/* ! E I G E N */


// funkcni jendoducha fce
int square(double x) {
    return x * x;
}


// soucet dvou np array
py::array_t<double> add_arrays(py::array_t<double> input1, 
							   py::array_t<double> input2) 
{
    auto buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.shape[0] != buf2.shape[0])
        throw std::runtime_error("Input shapes must match");

    auto result = py::array(py::buffer_info(
							nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
							sizeof(double),     /* Size of one item */
							py::format_descriptor<double>::value, /* Buffer format */
							buf1.ndim,          /* How many dimensions? */
							{ buf1.shape[0] },  /* Number of elements for each dimension */
							{ sizeof(double) }  /* Strides for each dimension */
						));

    auto buf3 = result.request();

    double *ptr1 = (double *) buf1.ptr,
           *ptr2 = (double *) buf2.ptr,
           *ptr3 = (double *) buf3.ptr;

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}



// FOR THE PYBIND11
// LEARN ABOUT py::vectorize()

// m.def .. definuje funkci
// m.def('square', &square)
// to znamena reference na funkci square o par radku vyse
PYBIND11_MODULE(somecode, m) {
    m.def("square", &square);
    m.def("add_arrays",  &add_arrays, "add two NumPy arrays");
    m.def("inv", &inv);
    m.def("det", &det);
}
