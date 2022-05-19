from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from flask import render_template
import pandas as pd
import math
from utils import make_html
import filetype
import requests
import itertools
from math import radians, degrees, sin, cos, asin, acos, sqrt

API_KEY = 'AIzaSyDA4dcO2U2_Bs2UFDC80bdSq54LmR247SM'
URL_PREFIX = 'https://maps.googleapis.com/maps/api/streetview/metadata?size=600x300&'
URL_SUFIX = '&fov=110&pitch=38&key='
label_map = {0: '106',
 1: '14',
 2: '15',
 3: '16',
 4: '17',
 5: '22',
 6: '23',
 7: '24',
 8: '25',
 9: '26',
 10: '27',
 11: '28',
 12: '29',
 13: '30',
 14: '33',
 15: '34',
 16: '35',
 17: '36',
 18: '37',
 19: '38',
 20: '39',
 21: '4',
 22: '40',
 23: '41',
 24: '42',
 25: '46',
 26: '47',
 27: '5',
 28: '58',
 29: '59',
 30: '69',
 31: '70',
 32: '81',
 33: '82',
 34: '94'}



def great_circle(lat_1, lon_1, lat_2, lon_2):
    #print(lon_1, lat_1, lon_2, lat_2)
    lon_1, lat_1, lon_2, lat_2 = map(radians, [lon_1, lat_1, lon_2, lat_2])
    
    return 6371 * (
        acos(sin(lat_1) * sin(lat_2) + cos(lat_1) * cos(lat_2) * cos(lon_1 - lon_2))
    )

def get_weights(w):
    weights = []
    for idx, i in enumerate(w):
        weights.append((label_map[idx], i))
    return weights

def get_closest_groups(group, gr_ref, n = 4):
    group_lat = gr_ref.loc[int(group)]['center_latitude']
    group_lon = gr_ref.loc[int(group)]['center_longitude']
    distances = dict()
    for idx in gr_ref.index:
        if int(group) != int(idx):
            distance = great_circle(group_lat, group_lon, gr_ref.loc[idx]['center_latitude'], gr_ref.loc[idx]['center_longitude'])
            distances[str(idx)] = distance

    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    values = distances.values()
    min_ = min(values)
    max_ = max(values)

    n_closest = dict(itertools.islice(distances.items(), 0, n))

    return n_closest

def get_street_view_coordinates(coordinates, radius = 5000):
    coords = str(coordinates[0]) + ',' + str(coordinates[1])
    res = requests.get(URL_PREFIX + 'location=' + coords + '&radius=' + str(radius) + URL_SUFIX + API_KEY)
    if res.json()['status'] == 'ZERO_RESULTS':
        return False
    return (res.json()['location']['lat'], res.json()['location']['lng'])

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config.update(
    TESTING=True,
    ENV='development',
    TEMPLATES_AUTO_RELOAD=True
)
model = None

def calculate_predicted_coordinates(weights, group_reference, n = 4, street_view = True):
    gr_ref = pd.read_csv(group_reference)
    gr_ref.dropna(inplace = True)
    groups = gr_ref['group']
    groups = list(map(str, list(map(int, groups))))
    gr_ref['group'] = groups
    result_longitude = 0
    result_latitude = 0
    top_weight = weights[0]
    closest = get_closest_groups(top_weight[0], gr_ref)

    new_probabilities = []
    for weight in weights:
        group = weight[0]

        if group in closest.keys() or group == top_weight[0]:

            new_probabilities.append([group, weight[1]])
    
    weight_sum = sum([i[1] for i in new_probabilities])
    factor = 1/weight_sum
    for j in range(len(new_probabilities)):
        new_probabilities[j][1] = new_probabilities[j][1] * factor

    for prob in new_probabilities:
        center_longitude = gr_ref[gr_ref.group == prob[0]]['center_longitude'].values[0]
        center_latitude = gr_ref[gr_ref.group == prob[0]]['center_latitude'].values[0]
        result_longitude += center_longitude * prob[1]
        result_latitude += center_latitude * prob[1]
    if street_view:
        street_view = get_street_view_coordinates((result_latitude, result_longitude), 10000)
        if street_view:
            print('Moved it.')
            return street_view
        
        
    return (result_latitude, result_longitude)  


#loading our trained model
def load_custom_model():
    global model
    model = [load_model("../models/B0_final-05-0.40.h5"), load_model("../models/B2-final-07-0.41.h5"), load_model("../models/MbNet_final_0.36.h5")]
    print(model)

def prepare_image(image, target):
    image = resize(image, target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home(name=None, longitude=None, latitude=None):
    if (not flask.request.args.get('longitude')):
        make_html()
    return render_template('home.html', name=name, longitude=flask.request.args.get('longitude'), latitude=flask.request.args.get('latitude'))

@app.route("/new.html")
def new(name=None):
    return render_template('new.html', name=name)

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        files = flask.request.files.getlist("files")
        long_true = flask.request.form.get('longInput', type=str)
        lat_true = flask.request.form.get('latInput', type=str)
        try:
            long_true = float(long_true)
            lat_true = float(lat_true)
        except:
            long_true = None
            lat_true = None
        num_imgs = len(files)
        longitude = 0
        latitude = 0
        counter = 1
        for image in files:
            print(type(image))
            if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return flask.redirect(flask.url_for('home'))
            image = image.read()
            image = Image.open(io.BytesIO(image))
            image.save('pic.jpg')
            img = open("./pic.jpg", "rb")
            image = img.read()
            img.close()
            image = imread(io.BytesIO(image))

            image = [prepare_image(image, target=(224, 224)), prepare_image(image, target=(260, 260)), prepare_image(image, target=(224, 224))]
            preds = [m.predict(i)for m, i in zip(model, image)]
            y_preds = np.array(preds)
            summed = np.sum(y_preds, axis = 0) / 3

            weights = get_weights(summed)
            weights.sort(key=lambda x:x[1], reverse = True)
            pred = calculate_predicted_coordinates(weights, "./group_reference_2.csv",street_view = False)

            counter += 1
            lon_i, lat_i = pred[0], pred[1]
            longitude += lon_i
            latitude += lat_i
        longitude /= num_imgs
        latitude /= num_imgs

    # return the data dictionary as a JSON response
    return flask.jsonify({
        "longitude"=str(longitude[0]),
        "latitude"=str(latitude[0])
    }
    )


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_custom_model()
    app.run()