import base64
import io
import json
import logging
import os
import time
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask.json import jsonify
from flask_cors import CORS
from PIL import Image

from main import SceneTextExtractor

app = Flask(__name__)
CORS(app)
logging.getLogger().setLevel(logging.INFO)

models = SceneTextExtractor()


def read_image(img_bytes):
    return cv2.imdecode(np.asarray(bytearray(img_bytes.read()), dtype="uint8"), cv2.IMREAD_COLOR)


@app.route("/", methods=["POST"])
def predict():
    start_time = time.time()
    response = {"success": False}
    try:
        image = request.files.get("image", None)

        if image is not None:
            image = read_image(image)
        elif request.data is not None:
            image = cv2.imdecode(np.fromstring(
                request.data, np.uint8), cv2.IMREAD_COLOR)
        res = models.process(image)
        response["prediction"] = res
        response["success"] = True
    except Exception as ex:
        response["ex"] = ex
        print(traceback.format_exc())
    response['run_time'] = "%.2f" % (time.time() - start_time)
    return jsonify(response)


@app.route("/layout", methods=["POST"])
def predict_layout():
    start_time = time.time()
    response = {"success": False}
    try:
        image = request.files.get("image", None)

        if image is not None:
            image = read_image(image)
        elif request.data is not None:
            image = cv2.imdecode(np.fromstring(
                request.data, np.uint8), cv2.IMREAD_COLOR)
        res = models.layout_model.process(image)
        response["prediction"] = res
        response["success"] = True
    except Exception as ex:
        response["ex"] = ex
        print(traceback.format_exc())
    response['run_time'] = "%.2f" % (time.time() - start_time)
    return jsonify(response)


@app.route("/ocr", methods=["POST"])
def predict_ocr():
    start_time = time.time()
    response = {"success": False}
    try:
        image = request.files.get("image", None)

        if image is not None:
            image = read_image(image)
        elif request.data is not None:
            image = cv2.imdecode(np.fromstring(
                request.data, np.uint8), cv2.IMREAD_COLOR)
        res = models.ocr_model.predict([image])
        response["prediction"] = res
        response["success"] = True
    except Exception as ex:
        response["ex"] = ex
        print(traceback.format_exc())
    response['run_time'] = "%.2f" % (time.time() - start_time)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)