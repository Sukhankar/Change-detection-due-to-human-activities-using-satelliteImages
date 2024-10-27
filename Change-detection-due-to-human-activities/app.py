from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import skimage.morphology
import uuid
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

app.config['UPLOAD_FOLDER'] = './static/output'

CORS(app)

def is_satellite_image(image_path):
    """ Basic check to determine if the image is likely a satellite image based on dimensions. """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Define minimum acceptable dimensions for satellite images
    min_height, min_width = 500, 500  # Example threshold, can be adjusted based on your needs

    if height < min_height or width < min_width:
        return False
    return True

def preprocess_satellite_image(image_path):
    print('[INFO] Preprocessing satellite image...')
    # Load satellite image with multiple bands (if available)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to RGB if the image has more than 3 channels (assuming multi-spectral bands)
    if image.shape[2] > 3:
        image = image[:, :, :3]  # Use only the first 3 bands (or select NIR based on the data)

    # Apply noise reduction or cloud masking if needed
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image

def geo_align_images(image1, image2):
    print('[INFO] Geo-aligning satellite images...')
    # Implement geo-alignment here if needed
    return image1, image2

def process_satellite_images(image1_path, image2_path, output_dir):
    print('[INFO] Reading and Preprocessing Satellite Images...')
    image1 = preprocess_satellite_image(image1_path)
    image2 = preprocess_satellite_image(image2_path)

    # Ensure Geo-Alignment of Satellite Images
    image1, image2 = geo_align_images(image1, image2)

    # Resize Images
    print('[INFO] Resizing Images...')
    new_size = (image1.shape[0] // 5 * 5, image1.shape[1] // 5 * 5)
    image1 = cv2.resize(image1, (new_size[1], new_size[0])).astype(int)
    image2 = cv2.resize(image2, (new_size[1], new_size[0])).astype(int)

    # Difference Image
    print('[INFO] Computing Difference Image...')
    diff_image = abs(image1 - image2)
    cv2.imwrite(os.path.join(output_dir, 'difference.jpg'), diff_image)
    diff_image = diff_image[:, :, 1]

    # PCA and Feature Vector Space
    print('[INFO] Performing PCA...')
    pca = PCA()
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca.fit(vector_set)
    EVS = pca.components_

    print('[INFO] Building Feature Vector Space...')
    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3

    # Clustering for Change Detection
    print('[INFO] Clustering for Change Detection...')
    least_index, change_map = clustering(FVS, components, new_size)
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)
    change_map_path = os.path.join(output_dir, 'ChangeMap.jpg')
    cv2.imwrite(change_map_path, change_map)

    # Morphological Operations
    print('[INFO] Performing Morphological Operations...')
    kernel = skimage.morphology.disk(6)
    CloseMap = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'CloseMap.jpg'), CloseMap)

    OpenMap = cv2.morphologyEx(CloseMap, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, 'OpenMap.jpg'), OpenMap)

    return change_map_path

def find_vector_set(diff_image, new_size):
    vector_set = []
    for j in range(0, new_size[0], 5):
        for k in range(0, new_size[1], 5):
            block = diff_image[j:j + 5, k:k + 5]
            if block.shape == (5, 5):
                feature = block.ravel()
                vector_set.append(feature)
    
    vector_set = np.array(vector_set)
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new_size):
    feature_vector_set = []
    for i in range(2, new_size[0] - 2):
        for j in range(2, new_size[1] - 2):
            block = diff_image[i-2:i+3, j-2:j+3]
            if block.shape == (5, 5):
                feature = block.flatten()
                feature_vector_set.append(feature)

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    return FVS

def clustering(FVS, components, new_size):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)
    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (new_size[0] - 4, new_size[1] - 4))
    return least_index, change_map

@app.route('/process', methods=['POST'])
def process():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "No file part"}), 400

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    if image1_file.filename == '' or image2_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image1_file and image2_file:
        # Save files locally temporarily
        image1_filename = str(uuid.uuid4()) + "_" + image1_file.filename
        image2_filename = str(uuid.uuid4()) + "_" + image2_file.filename
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1_filename)
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2_filename)
        image1_file.save(image1_path)
        image2_file.save(image2_path)

        # Check if the uploaded images are satellite images
        if not is_satellite_image(image1_path) or not is_satellite_image(image2_path):
            os.remove(image1_path)
            os.remove(image2_path)
            return jsonify({"error": "Both images must be satellite images"}), 400

        change_map_path = process_satellite_images(image1_path, image2_path, app.config['UPLOAD_FOLDER'])

        # Remove local files
        os.remove(image1_path)
        os.remove(image2_path)

        return jsonify({
            "message": "Processing complete",
            "change_map_path": change_map_path
        })

    return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True)
