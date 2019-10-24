# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages

import PIL
from PIL import Image,ImageChops
#from Pillow import Image,ImageChops
from PIL.ExifTags import TAGS
#from Pillow.ExifTags import TAGS
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter import filedialog
#import seaborn as sns
import os
import cv2
#from DateTime import DateTime
#from datetime import datetime
#import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
#import scipy
#from scipy.ndimage.filters import gaussian_filter
root = tkinter.Tk()

# depencencies for stitcher
#from pyimagesearch.panorama import Stitcher
#import argparse
import imutils

"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
    help="path to the first image")
ap.add_argument("-s", "--second", required=True,
    help="path to the second image")
args = vars(ap.parse_args())
"""



def get_directory_name(caption):
    dirname = tkinter.filedialog.askdirectory(parent=root,initialdir="/",title=caption)  
    if len(dirname ) > 0:
        print (' You chose %s' % dirname)
    return dirname

def get_file_name(caption):
    file = tkinter.filedialog.askopenfile(parent=root,mode='rb',title=caption)
    if file != None:
        data = file.read()
    #file.close()
    print (' I got %d bytes from this file.' % len(data))
    return file

def save_file_name(caption):
    myFormats = [
    ('Windows Bitmap','*.bmp'),
    ('Portable Network Graphics','*.png'),
    ('JPEG / JFIF','*.jpg'),
    ('CompuServer GIF','*.gif'),
    ]
    fileName = tkfiledialog.asksaveasfilename(parent=root,filetypes=myFormats ,title=caption)
    if len(fileName ) > 0:
        print ('Now saving under %s' % nomFichier)

    return fileName

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
		self.isv4 = imutils.is_cv4()
		#print('self.isv3',self.isv3,'self.isv4',self.isv4)

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3 or self.isv4:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			print('H \n',H)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		print ('No matches could be established')
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis

def stitch_images():
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)

    # Get first file to process
    print ('Please Browse for first image to be stitched')
    ImageFile = get_file_name('Select first (left) image file')
    img1 = ImageFile.name
    imageA = cv2.imread(img1)

    # Get second file to process
    print ('Please Browse for second image to be stitched')
    ImageFile = get_file_name('Select second (right) image file')
    img2 = ImageFile.name
    imageB = cv2.imread(img2)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    # Plot result
    # show the images

    # Do you want to show input images?
    Yes_No_string = str(input ('Do you waant to show input images (y/n) '))
    if Yes_No_string == 'y' or Yes_No_string == 'Y':
        cv2.imshow("Image A", imageA)
        cv2.imshow("Image B", imageB)
    # Do you want to show matches?
    Yes_No = False
    Yes_No_string = str(input ('Do you waant to show matches (y/n) '))
    if Yes_No_string == 'y' or Yes_No_string == 'Y':
        cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
 


def main():
    #matching_with_ORB()
    stitch_images()

    return

main()
"""
if __name__ == "main":
     # execute only if run as a script
     main()
"""   
