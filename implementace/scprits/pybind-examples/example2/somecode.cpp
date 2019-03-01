/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(double x) {
    return x * x;
}


// m.def .. definuje funkci
// m.def('square', &square)
// to znamena reference na funkci square o par radku vyse
PYBIND11_MODULE(somecode, m) {
    m.def("square", &square);
}
