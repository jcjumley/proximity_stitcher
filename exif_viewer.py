"""
exif writer:
    (1) opens a file:
    (2) read/priont its exif data
    (2) changes it's name
    (3) edit's exif data
    (4) saves file with new name and edited exif data
"""
# import the necessary packages
import piexif
import piexif.helper
import PIL
from PIL import Image,ImageChops

import tkinter
from tkinter import *
from tkinter import filedialog
import os

from DateTime import DateTime
from datetime import datetime
import math

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

def problem_1():

    # Get first file to process
    print ('Please Browse  image ')
    ImageFile = get_file_name('Select Image File')
    print("you selected: ",ImageFile)
    im = PIL.Image.open(ImageFile)    

    # Read exif data
    #exif_dict = piexif.load(ImageFile.name)
    exif_dict = piexif.load(im.info["exif"])
##
##    for ifd_name in exif_dict:
##        if ifd_name == "thumbnail":  # Don't print the thumbnail
##            break
##        print("\n{0} IFD:".format(ifd_name))
##        for key in exif_dict[ifd_name]:
##            try:
##                print(key, exif_dict[ifd_name][key][:10])
##            except:
##                print(key, exif_dict[ifd_name][key])

    try:
        print('\n User Comment ' + str(exif_dict["Exif"][piexif.ExifIFD.UserComment]))
    except:
        pass
    return
    

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
