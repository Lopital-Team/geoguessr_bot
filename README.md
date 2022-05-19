# geoguessr_bot
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is simple Lorem ipsum dolor generator.
	
## Technologies
Project is created with:
* Python
* Tensorflow
* Keras
* Pandas
* Numpy
* pytesseract
* OpenCV
	
## Setup
To preprocess the data, train and evaluate the models, you can do so all within the project_notebook.ipynb

Testing can be done within the web application inside the web_app directory.
Running the web application should be done like so:

```
$ cd ./web_app
$ python app.py
```

Alternatively, you can run a simpler version within the rest_api directory like so:
```
$ cd ./rest_api
$ python app.py
```
This way, you can send a post request with the images and the output will be the predicted coordinates for that specific location.
