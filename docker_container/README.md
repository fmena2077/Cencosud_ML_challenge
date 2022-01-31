# Docker Container for predicting Meat Sales

To build the docker image:

docker build -t < image name > . 

To run the image:

docker run -e "DATE=< date value >" < image name >

Example:

docker build -t meatsales .

docker run -e "DATE=2019-12-13" meatsales 


