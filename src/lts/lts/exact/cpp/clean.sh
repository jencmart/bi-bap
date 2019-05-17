#!/bin/bash

rm *.cpython*.so  # binary for import
rm .*.cpp.cppimporthash # hash to check if source sode is different
rm .rendered.*.cpp # source code without cppimort directives, I don't know why cppimport create this

# python3 ./test.py
