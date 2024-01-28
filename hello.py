import numpy as np
from flask import Flask, request, render_template, make_response
from flask_restful import Resource, Api
import cv2
import urllib.request
from io import BytesIO

app = Flask(__name__, template_folder='templates')
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class NaTrzy(Resource):

    def get(self):
        img = cv2.imread('family.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return {'count': len(boxes)}


class NaCztery(Resource):
    def get(self):
        # Get the image URL from the query parameters
        image_url = request.args.get('url')
        response_get = urllib.request.urlopen(image_url)
        img_bytes = BytesIO(response_get.read())
        img = cv2.imdecode(np.frombuffer(img_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return {'count': len(boxes)}


class NaPiec(Resource):
    def get(self):
        return make_response(render_template('upload_form.html'))

    def post(self):
        photo = request.files['photo'].read()
        img = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return {'count': len(boxes)}


api.add_resource(NaTrzy, '/3')
api.add_resource(NaCztery, '/4')
api.add_resource(NaPiec, '/5')

if __name__ == '__main__':
    app.run(debug=True)
