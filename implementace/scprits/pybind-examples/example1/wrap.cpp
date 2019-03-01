#include <pybind11/pybind11.h>
#include "funcs.hpp"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(wrap, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers",
          "i"_a=1, "j"_a=2);

}
// pybind11.get_include()
// /usr/local/include/python3.5
// /usr/local/include/python3.5/pybind11/pybind11.h


