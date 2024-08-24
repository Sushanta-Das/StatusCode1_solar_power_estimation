from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)
@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json

    # Decode the image
    img_data = data.get('image')
    if not img_data:
        return jsonify({"error": "No image provided"}), 400

    # Assuming the image is in base64 format
    img_data = img_data.split(",")[1]  # Remove the data:image/png;base64, part
    img_bytes = base64.b64decode(img_data)

    # Open the image
    image = Image.open(BytesIO(img_bytes))
   
    # Perform operations on the image
    # For demonstration, we'll calculate the percentage of non-white pixels as the "percent" of the building
    np_image = np.array(image)
    print(np_image)
    # Simple threshold to count non-white pixels (adjust threshold according to your needs)
    non_white_pixels = np.count_nonzero(np_image < 250)
    total_pixels = np_image.size / np_image.shape[2]
    percent = non_white_pixels / total_pixels

    # Return the result
    return jsonify({
        "percent": percent
    })

if __name__ == '__main__':
    app.run(debug=True)
