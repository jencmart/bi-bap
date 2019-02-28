# Quick overview
* Implementation of FAST-LTS by [2]- done (python); (working on C implementation)
* nothing else yet

# Task overview
The least trimmed squares (LTS) method is a robust version of the classical method of least squares used to find an estimate of coefficients in the linear regression model. Computing the LTS estimate is known to be NP-hard, and hence suboptimal probabilistic algorithms are used in practice.

1) Describe robust regression methods and give a detail description of the LTS method.
2)Survey known algorithms for computing the LTS estimate.
3) Create a generator of datasets enabling to set parameters like data size, contamination etc.
4) Implement selected algorithms use these datasets to compare their performance


[1] J. Agulló, New algorithms for computing the least trimmed squares regression estimator, Computational Statistics &amp; Data Analysis, v.36 n.4, p.425-439, June 28 2001.
Zadost na ResearchGate

[2] D. M. Hawkins, The feasible solution algorithm for least trimmed squares regression, Computational Statistics &amp; Data Analysis, v.17 n.2, p.185-196, Feb. 1994.

[3] P.J. Rousseeuw, K. Van Driessen, Technical Report, University of Antwerp, 1999.
	Precteno - podle toho implementovano

[4] Karel Klouda. An exact polynomial time algorithm for computing the least trimmed squares estimate. Comput. Stat. Data Anal. 84, C (April 2015), 27-40.



# python Environment - docker container with Anaconda

* change your src directory accordingly !
* optionally change container name :-)
```
# first time
docker pull continuumio/anaconda3

docker run -it --name python-anaconda3 --mount type=bind,source="/home/jencmart/fit/",target=/opt/notebooks -p 8888:8888 continuumio/anaconda3 /bin/bash 

# next time
docker start -i python-anaconda3
```

```
# inside container
cd /opt/notebooks
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# and if you are not afraid you can disable token and password with (== don't do this)
 --NotebookApp.token='' --NotebookApp.password=''
```

* outside container you have jupyter notebook at
	* http://127.0.0.1:8888

#### And if you want to automate whole process
```
# inside container
echo 'cd /opt/notebooks/' >> .bashrc
echo 'jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root' >> .bashrc
```
* With this setup you don't need to even go inside the container, so you cant just start it detached

```
docker start python-anaconda3
```

* if you just want to run some pytohn scripts, feel free to do it within the container

```
docker start -i python-anaconda3

python /path/to/my/scrpit
```

# python -> pybind11 -> eigen
```
pip install pybind11
pip install cppimport
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror
mkdir example-program
cd example-program
touch funcs.hpp funcs.cpp wrap.cpp setup.py test_funcs.py

```

# Teoretical part
* Pouzit sablony latex
* compiler - texlive
* gui - gummi

