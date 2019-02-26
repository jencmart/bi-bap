## Rychly prehled
* Implementace algoritmu FAST-LTS hotovo (python), i s generatorem dat, na prepsani do C se pracuje
* to je zatim vse


## python Environment - docker container with Anaconda

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
```

* outside container you have jupyter notebook at
	* http://127.0.0.1:8888


* if you just want to run some pytohn scripts, feel free to do it within the container
