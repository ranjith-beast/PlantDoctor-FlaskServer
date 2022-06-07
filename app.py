from asyncio import wait

import tensorflow as tf
from PIL import ImageOps, Image
from flask import Flask, request
import json
import numpy as np
import cv2
from flask_cors import CORS
import pandas as pd
from sklearn import linear_model

# dataset = pd.read_csv("apy.csv")
# dataset = dataset.dropna()
# regg=linear_model.LinearRegression()
np.set_printoptions(suppress=True)
leafdetection_model = tf.keras.models.load_model('predictleafmodel\keras_model.h5')
print("Done loading leafdetection_model")
with open("Categories.json") as f:
    name = json.load(f)
with open("labels.json") as l:
    object_name = json.load(l)
# print(data)
model = tf.keras.models.load_model("Xception Model")
print('done')
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        print(type(file))
        print(file.filename)
        file.save(file.filename)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file.filename)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = leafdetection_model.predict(data)
        leaf_or_other = np.argmax(prediction)
        thing = object_name[leaf_or_other]
        print(thing)
        if thing == "Other":
            return {"disease": "Other"}
        image = cv2.imread(file.filename)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, 128, 128, 3)
        img = img.astype('float32')
        img = img / 255.0
        disease_prediction = model.predict(img)
        class_names = np.argmax(disease_prediction)
        disease_name = name[class_names]
        print(disease_name)
        return {"disease": disease_name}

# @app.route('/getProduction',methods=['GET','POST'])
# def getProduction():
#     if request.method == 'GET':
#         print(request)
#         district = request.args.get('district')
#         crop = request.args.get('crop')
#         print(district)
#         print(crop)
#         x = []
#         y = []
#         for row in dataset.iterrows():
#             if row[1][1] == district and row[1][4] == crop:
#                 x.append(row[1][5])
#                 print(x)
#                 y.append(row[1][6])
#                 print(y)
#         # x = dataset[dataset['Crop'] == crop]['Area']
#         # y = dataset[dataset['Crop'] == crop]['Production']
#         x = dataset[dataset['Crop'] == crop]['Area']
#         y = dataset[dataset['Crop'] == crop]['Production']
#         train = pd.Series(x)
#         test = pd.Series(y)
#         regg.fit(train.values.reshape(-1, 1), test.values.reshape(-1, 1))
#         coeff = regg.coef_.tolist()
#         print(coeff)
#         return {"prediction": coeff}

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
        Wrong URL!
        <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
        An internal error occurred: <pre>{}</pre>
        See logs for full stacktrace.
        """.format(e), 500


# @app.route('/api', methods=['POST'])
if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('0.0.0.0', 5000, app)
