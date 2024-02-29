Python 3.10.5 (tags/v3.10.5:f377153, Jun  6 2022, 16:14:13) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import cv2
import numpy as np

def stitch_images(image1, image2, overlap_percentage):
    # Load images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Resize images to have the same height
    height, width = img1.shape[:2]
    new_width = int(width / (1 + overlap_percentage))
    img1 = cv2.resize(img1, (new_width, height))
    img2 = cv2.resize(img2, (new_width, height))

    # Find keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Brute Force Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Homography
    if len(good_matches) > 10:
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
