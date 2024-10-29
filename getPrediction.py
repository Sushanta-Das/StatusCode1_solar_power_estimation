from keras.saving import register_keras_serializable
from keras.models import load_model

import numpy as np
import tensorflow as tf
import cv2 

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
    model = load_model(model_path, custom_objects={'ResizeLayer': ResizeLayer})   

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

prediction =get_prediction('multipleRoof.png','unet_rooftop_model1.keras')
print(np.count_nonzero(prediction == 1))