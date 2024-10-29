from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np

import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.models import load_model
import cv2
import requests
from flask_cors import CORS

import matplotlib.pyplot as plt

def visualize_predictions(images, predictions):
    # Adjusting the figure size to make the images larger
    plt.figure(figsize=(15, len(images) * 6))
    IMG_HEIGHT, IMG_WIDTH = 256, 256
    for i in range(len(images)):
        # Display the original image
        plt.subplot(len(images), 2, i*2 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')  # Hide axes for better visualization

        # Display the predicted mask
        plt.subplot(len(images), 2, i*2 + 2)
        plt.imshow(predictions[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')  # Hide axes for better visualization

    plt.tight_layout()  # Adjust subplots to fit into figure area
    plt.show()

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

def preprocess_image(image, img_height=256, img_width=256):
    """Preprocess the image for model prediction."""
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    if image.ndim == 2:  # Convert grayscale to RGB
        image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
    image_resized = tf.image.resize(image, (img_height, img_width))
    image_normalized = image_resized / 255.0
    return image_normalized

def get_prediction(image, model):
    """Generate predictions from the model on the processed image."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.array([preprocessed_image]))[0]
    prediction=(prediction > 0.5).astype(np.uint8)
    
    # Convert prediction to binary mask if necessary
    return prediction

# Load the model and add custom objects
model_path = "unet_rooftop_model1_revised.keras"  # Replace with actual model path
model = load_model(model_path, custom_objects={'ResizeLayer': ResizeLayer})

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get the image file and additional data from the request
    image_file = request.files.get('image')
    long = request.form.get('long')
    lat = request.form.get('lat')
    area_rect = float(request.form.get('area', 0))  # Convert area to float

    # Open and process the image for model prediction
    image = Image.open(image_file.stream)
    np_image = np.array(image)

    # Perform model prediction
    building_prediction = get_prediction(np_image, model)

    # Calculate percentage of non-white pixels for building area
    white_pixelcount = np.sum(building_prediction > 0.5)  # Adjust as needed
    print(white_pixelcount)
    total_pixelcount = np_image.size
    ratio_white_pixel = white_pixelcount / total_pixelcount
    area_building = area_rect * ratio_white_pixel

    # Fetch solar radiation data
    response = requests.get(
        f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=SB&longitude={long}&latitude={lat}&start=20230101&end=20230101&format=json"
    )
    solar_data = response.json()
    solar_radiation_per_unit_area = solar_data['properties']['parameter']['ALLSKY_SFC_SW_DWN']['20230101']
    power = solar_radiation_per_unit_area * area_building

    # Return the calculated power in the response
    return jsonify({
        "power": power
    })

if __name__ == '__main__':
    app.run(debug=True)
