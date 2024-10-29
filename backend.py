from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np

import tensorflow as tf
import cv2 
import requests
from flask_cors import CORS, cross_origin

@register_keras_serializable()
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_shape)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
def get_prediction(image_path, model_path):
   

    # Define the image dimensions used during training
    IMG_HEIGHT, IMG_WIDTH = 256, 256

    def preprocess_image(image):
        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.ndim == 2:  # Grayscale image
            image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
        image_resized = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image_normalized = image_resized / 255.0
        return image_normalized

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.array([preprocessed_image]))

    # Convert prediction to binary mask (optional, depending on your use case)
    prediction = (prediction > 0.5).astype(np.uint8)
    # visualize_predictions([image], [prediction])
    return prediction
    
model = load_model(model_path, custom_objects={'ResizeLayer': ResizeLayer})   

app = Flask(__name__)
cors = CORS(app)
@app.route('/upload', methods=['POST'])

def upload_image():
    # Get the image from the form data
    image = request.files['image']
    #store long , lat, area of reactangle
    long = request.form['long']
    lat = request.form['lat']
    area_rect = request.form['area']
    #print(request.files)
    print(request.files)
    # Open the image
    image = Image.open(image.stream)

    # Perform operations on the image
    # For demonstration, we'll calculate the percentage of non-white pixels as the "percent" of the building
    np_image = np.array(image)
    # print(np_image)
    # Simple threshold to count non-white pixels (adjust threshold according to your needs)
    #model image output
    white_pixelcount = np.sum(np_image > 150)
    total_pixelcount = np_image.size
    ratio_white_pixel = white_pixelcount / total_pixelcount
    area_building = float(area_rect) * ratio_white_pixel
    getsolarRadiation__per_unit_area=requests.get(f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=SB&longitude={long}&latitude={lat}&start=20230101&end=20230101&format=json")
    solarRadiation__per_unit_area=getsolarRadiation__per_unit_area.json()
    power = solarRadiation__per_unit_area['properties']['parameter']['ALLSKY_SFC_SW_DWN']['20230101']*area_building
    print(power)
    # Return the result
    return jsonify({
        "power": power
    })

if __name__ == '__main__':
    app.run(debug=True)
