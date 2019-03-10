# PIBYND 11 + cppimport + setuptools

## Prerequisities
* python
* setuptools
* pybind11
* cppimport # it is not inside conda, need to install it through pip
* eigen headers in ./src/lib/eigen


## Development
* edit files in ./src/devel/src
* somecode.cpp - c++ code with cppimport anotation
* uses eigen library located in ./src/lib/eigen
* test.py - uses cppimport to autocompile and import somecode.cpp
* for cleanup purposes you can can use run.sh
* test-from-notebook - alternative to test.py, but it is better to use installed code

* somefile.so (compiled code) will be created after cppimport compile the code, together with sum checksums (hidden files)
    * feel free to delete it (test.py will complie it again)

### TL`DR
* python3  ./src/devel/src/test.py
* or ./run.sh (does the same)


## Deployment
* use ./setup.py
* pip3 install ./
* it'll use code from ./src/somecode.cpp
* note that it is only simlink to ./src/devel/src/somecode.cpp
* check if path to eigen library is correctly set up in ./setup.py
	* expected path './src/lib/eigen'
* when you install package in some environment, you can simply import it as package
* see ./tests/test.py

* if you edit the code in /.src/devel/src/somecode.cpp and want to reintall the packeage, copy it to ./src/somecode.cpp and remove cppimport header ?

### TL`DR
* pip install ./setup.py

