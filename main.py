import os

from keras.models import load_model
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify

app = Flask("__name__")

model = load_model("Braintumormodel.h5")
model.make_predict_function()
IMG_SIZE = 200


@app.route("/", methods=['get'])
def home():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "something went wrong 1"

        user_file = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."

        else:
            user_file = request.files['file']
            path = os.path.join(os.getcwd() + user_file.filename)
            user_file.save(path)
            prediction = model.predict([prepare(path)])

            return jsonify({
                "status": "success",
                "prediction": predictPicture(prediction)
            })


def prepare(filepath):
    img_array = cv2.imread(filepath)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def predictPicture(prediction):
    if np.argmax(prediction) == 0 or np.argmax(prediction) == 1 or np.argmax(prediction) == 3:
        return True
    else:
        return False


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
