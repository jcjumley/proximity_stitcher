Python 3.10.5 (tags/v3.10.5:f377153, Jun  6 2022, 16:14:13) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
python
Copy code
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

def extract_features(image_path):
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features

def match_features(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

def stitch_images(image1, image2, overlap_percentage):
    # Extract features using TensorFlow
    features1 = extract_features(image1)
    features2 = extract_features(image2)

    # Match features
    good_matches = match_features(features1, features2)

    if len(good_matches) > 10:
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        return result
    else:
        print("Not enough matches are found.")
        return None

# Example usage
image1_path = "path/to/image1.jpg"
image2_path = "path/to/image2.jpg"
overlap_percentage = 0.2  # Adjust this based on your requirements

result_image = stitch_images(image1_path, image2_path, overlap_percentage)

if result_image is not None:
    cv2.imshow("Stitched Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image stitching failed.")