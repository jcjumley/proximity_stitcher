"""
exif writer:
    (1) opens a file:
    (2) changes it's name
    (3) reads and edit's exif data
    (4) saves vile with new name
"""
# import the necessary packages
import piexif
import PIL
from PIL import Image,ImageChops
#from Pillow import Image,ImageChops
from PIL.ExifTags import TAGS
#from Pillow.ExifTags import TAGS
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter import filedialog
import seaborn as sns
import os
import cv2
from DateTime import DateTime
from datetime import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import scipy
from scipy.ndimage.filters import gaussian_filter

root = tkinter.Tk()

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

# gets the file meta data for an image file
def get_exif(fn):
    ret = {}
    i = PIL.Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

# gets gps data in decimal degrees from image meta data
def get_gps_deg(exf):
    GPS = []
    gps = exf['GPSInfo']
    GPS0 = float(gps[2][0][0])/float(gps[2][0][1]) + float(gps[2][1][0])/(60.*float(gps[2][1][1])) + float(gps[2][2][0])/(3600*float(gps[2][2][1]))
    if (gps[1] == u'S'):
        GPS0 = -GPS0 
    GPS.append(GPS0)
    GPS1 = float(gps[4][0][0])/float(gps[4][0][1]) + float(gps[4][1][0])/(60.*float(gps[4][1][1])) + float(gps[4][2][0])/(3600*float(gps[4][2][1]))
    if (gps[3] == u'W'):
        GPS1 = -GPS1
    GPS.append(GPS1)
    GPS2 = float(gps[6][0])/float(gps[6][1])
    GPS.append(GPS2)
    return GPS

def problem_1():

    # Get first file to process
    print ('Please Browse  image ')
    ImageFile = get_file_name('Select Image File')

    

    # Get output file directory and file name
    print ('browse for directory to store output file')
    output_Dir = get_directory_name('select output directory')
    output_filename = input(' Enter output file name: ')
    output_file = output_Dir + '/' + output_filename + '.JPEG'

    # Read exif data
    img1 = PIL.Image.open(ImageFile)

    exif_dict = piexif.load(img1.info["exif"])
    w, h = img1.size
    exif_dict["0th"][piexif.ImageIFD.XResolution] = (w, 1)
    exif_dict["0th"][piexif.ImageIFD.YResolution] = (h, 1)

    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag in exif_dict[ifd]:
            print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])

    exif_dict['New_Key'] = 999
    print("\n exif_dict['New_Key']",exif_dict['New_Key'])

    #exif_dict['GPSAltitude'] = (341424, 1000)
    exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (140, 1)
    print("\n ******exif_dict['GPSAltitude******']",exif_dict['GPSAltitude'][piexif.GPSIFD.GPSAltitude])
    
    exif_bytes = piexif.dump(exif_dict)
    
    #piexif.insert(exif_bytes,output_file)
    img1.save(output_file,exif=exif_bytes)
    img1.close()

    #Read copied file's metadata

    img2 = PIL.Image.open(output_file)
    exif_dict_2 = piexif.load(img2.info["exif"])
    w, h = img2.size
    exif_dict_2["0th"][piexif.ImageIFD.XResolution] = (w, 1)
    exif_dict_2["0th"][piexif.ImageIFD.YResolution] = (h, 1)

    print('\n ##### revised metadata #####')

    #for ifd in ("0th", "Exif", "GPS", "1st"):
    for ifd in (exif_dict_2):
        try:
            for tag in exif_dict_2[ifd]:
                print(piexif.TAGS[ifd][tag]["name"], exif_dict_2[ifd][tag])
        except:
            pass

    img2.close()
    

def main():
    problem_1()
    #problem_2()

    return

main()
"""
if __name__ == "main":
     # execute only if run as a script
     main()
"""
