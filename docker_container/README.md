# Docker Container for predicting Meat Sales

1. To build the docker image:

docker build -t < image name > . 

2. To run the image:

docker run -e "DATE=< date value >" < image name >

Example:

docker build -t meatsales .

docker run -e "DATE=2019-12-13" meatsales 


3. To save results on local machine:
docker run -e "DATE=2019-12-13" -v < absolute path to where you want to save >:< WORKDIR in container > meatsales

Example:
docker run -e "DATE=2019-12-13" -v C:\Users\franc\projects\letstf2gpu\Cencosud\docker_container:/usr/app/src meatsales
