from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import piexif
import piexif.helper
import PIL
from PIL import Image,ImageChops
from matplotlib.path import Path
import matplotlib.patches as patches

from PIL.ExifTags import TAGS
#import Tkinter, tkFileDialog
import tkinter
from tkinter import *
from tkinter import filedialog
import os
from DateTime import DateTime
import datetime
import pandas as pd
import numpy as np
import math
import csv
import sys

sys.path.insert(0,'C:\F_archive\easystore\Python_Programs')

import my_modules as my_modules
from my_modules import get_directory_name as get_directory_name
from my_modules import get_file_name as get_file_name
from my_modules import save_file_name as save_file_name
from my_modules import get_exif as get_exif
from my_modules import get_gps_deg as get_gps_deg
from my_modules import get_r_of_phi as get_r_of_phi
from my_modules import calculate_distance as calculate_distance
from my_modules import date_to_nth_day as date_to_nth_day
from my_modules import calculate_sun_angle as calculate_sun_angle
from my_modules import build_boundray as build_boundray
from my_modules import build_boundray_star as build_boundray_star

root = tkinter.Tk()
FOV = 78.8 #Mavic Pro camer field of view 78.8 Deg
EARTH_MEAN_RADIUS  =   6371.01	# In km
ASTRONOMICAL_UNIT  =  149597890	# In km

"""
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)
Image_A
Image_B
Ma
Mb
altitude_bias_left_image
altitude_bias_right_image
exif_A
exit_B
GPS_A
GPS_B
test
pixel_size_A
pixel_size_A
h_image_A
w_image_A
h_image_B
w_image_B
bb_A
bb_B
cov_A
cov_B
A_cropped
B_cropped
e_loss

# Construct a `Session` to execute the graph.
sess = tf.compat.v1.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
"""
print('day',date_to_nth_day(2019/8/30))

def stitch_images():
    # Get first file to process
    print ('Please Browse for first image to be stitched')
    ImageFile_A = get_file_name('Select first (left) image file')
    img1 = ImageFile_A.name
    Image_A = PIL.Image.open(img1)
    altitude_bias_left_image = 300

    # Get second file to process
    print ('Please Browse for second image (right) to be stitched')
    ImageFile_B = get_file_name('Select second image file')  
    img2 = ImageFile_B.name
    Image_B = PIL.Image.open(img2)
    altitude_bias_right_image = 300

    path_A = build_boundray(Image_A)
    print('path_A \n',path_A)
    print(path_A.contains_point((2000.,2000.)))
    print(Path.contains_point(path_A,(4000.,3000.)))
    path_B = build_boundray(Image_B)
    verts_A = Path.vertices
    print("verts_A \n",verts_A)

    # get gps data from left image
    exf_A = get_exif(img1)
    GPS_A = get_gps_deg(exf_A)
    height_left_image = GPS_A[2] - altitude_bias_left_image

    # get gps data from right image
    exf_B = get_exif(img2)
    GPS_B = get_gps_deg(exf_B)
    height_right_image = GPS_B[2] - altitude_bias_right_image

    # calculate distance between images from GPS data
    test = calculate_distance(GPS_A,GPS_B) # Distance camera position for Imsage A to image B (feet)
    print("offset distance between images = {0:5.2f} {1:5.2f} feet"
          .format(test[0],test[1]))

    # calculate pixel size (left image)
    Image_A_width = 2.0*height_left_image*math.tan((math.pi/180.)*FOV/2.0)
    Image_A_width = (26./35.)*Image_A_width
    (w_Image_A, h_Image_A)   = Image_A.size
    
    if (Image_A_width != 0.):
       pixel_size_A = abs(Image_A_width/float(max(w_Image_A,h_Image_A)))
    print ("pixels size A {0:8.5f} feet".format(pixel_size_A))

    # calculate pixel size (right image)
    Image_B_width = 2.0*height_left_image*math.tan((math.pi/180.)*FOV/2.0)
    Image_B_width = (26./35.)*Image_B_width
    (w_Image_B, h_Image_B)   = Image_B.size

    if (Image_B_width != 0.):
        pixel_size_B = abs(Image_B_width/float(max(w_Image_B,h_Image_B)))
    print ("pixels size B {0:8.5f} feet".format( pixel_size_B))

    #Ma is idenity matrix but could include GPS error for Image A
    Ma = np.identity(3,dtype=np.float64)

    # Build translation matrix Mb (first estimate)
    Mb = np.identity(3,dtype=np.float64)
    try:
        Mb[0][2] = -test[1]/pixel_size_B  # + 20
        Mb[1][2] = test[0]/pixel_size_B # - 645      # for debugging purposes added 10
    except:
        Mb[0][2] = 0.
        Mb[1][2] = 0.
    print ("Mb:")
    print ("[{0:12.5f} , {1:12.5f}, {2:12.5f}]"
           .format(Mb[0][0],Mb[0][1],Mb[0][2]))
    print ("[{0:12.5f} , {1:12.5f}, {2:12.5f} ]"
           .format(Mb[1][0],Mb[1][1],Mb[1][2]))
    print ("[{0:12.5f} , {1:12.5f}, {2:12.5f} ]"
           .format(Mb[2][0],Mb[2][1],Mb[2][2]))
           

    path_C = build_boundray_star(Image_B,Mb)
    print("path C \n",path_C)
           

    return (Image_A,Image_B)
def main():
    stitch_images()
    return

main()
if __name__ == "main":
    # execute only if run as a script
    main()
