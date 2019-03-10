# PIBYND 11 + cppimport + setuptools

## 0. For new packages

#### for cppimport chage
* filename `.src/xxx.cpp`
* inside   `.src/xxx.cpp` change `PYBIND11_MODULE(xxx, m)`
* inside   `./tests/test-dev.py` change `mujpackage = cppimport.imp("../src/xxx")`

#### for setuptools change also
* inside `./tests/test-setup.py` change `import xxx`
* inside `./setup.py`change package & module name to `xxx`


## 1. Prerequisities
* python
* setuptools
* pybind11
* cppimport # it is not inside conda, need to install it through pip
* eigen headers in ./lib/eigen # ./lib/eigen/..all..the..files..
* right now conda.recipe is unused...

## 2. Development
* edit files in `./src/xxx.cpp`
* fastlts.cpp - c++ code with cppimport anotation
* uses eigen library located in `./lib/eigen`
* ./tests/test-dev.py - uses cppimport to autocompile and import somecode.cpp
* for cleanup purposes you can can use `./src/clean.sh`

* fastlts.so (compiled code) will be created after cppimport compile the code, together with sum checksums (hidden files)
    * feel free to delete it (`./tests/test-dev.py will complie it again`)

### TL`DR
* python3  `./tests/test-dev.py`

### Debug
* all these three names needs to be the same say `xxx`
	* `./src/xxx.cpp`
		* inside this file `PYBIND11_MODULE(xxx, m)`
	* `./tests/test-dev.py`
		* `mujpackage = cppimport.imp("../src/xxx")`

## 3. Deployment
* to install module to the python environment use `./setup.py`
* `pip3 install ./`
* it'll use code from `./src/xxx.cpp`
* don't worry about cppimort annotation - it's basically a comment
* check if path to eigen library is correctly set up in `./setup.py`
	* expected path `./lib/eigen`
* when you install package in some environment, you can simply import it as package
* see `./tests/test-setup.py`

* you've edited the code in `/.src/xxx.cpp` and want to reinstall the package? simply run `* pip3 install ./` again

### TL`DR
* pip3 install ./

### Debug
* name of the module in `./setup.py` needs to be the same as in pybind annotation on `./src/xxx.cpp` so check that if some import errors are happening
