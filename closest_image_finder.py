from PIL import Image
from PIL.ExifTags import TAGS
import Tkinter, tkFileDialog
import os
from DateTime import DateTime
from datetime import datetime
import pandas as pd
import numpy
import math
import csv

root = Tkinter.Tk()
FOV = 78.8 #Mavic Pro camer field of vies 78.9 Deg
EarthMeanRadius  =   6371.01	# In km
AstronomicalUnit  =  149597890	# In km

def get_directory_name():
    dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
    if len(dirname ) > 0:
        print ' You chose %s' % dirname
    return dirname

def get_file_name():
    file = tkFileDialog.askopenfile(parent=root,mode='rb',title='Choose a file')
    if file != None:
        data = file.read()
    #file.close()
    print " I got %d bytes from this file." % len(data)
    return file

def save_file_name():
    myFormats = [
    ('Windows Bitmap','*.bmp'),
    ('Portable Network Graphics','*.png'),
    ('JPEG / JFIF','*.jpg'),
    ('CompuServer GIF','*.gif'),
    ]
    fileName = tkFileDialog.asksaveasfilename(parent=root,filetypes=myFormats ,title="Save the image as...")
    if len(fileName ) > 0:
        print "Now saving under %s" % nomFichier

    return fileName

# gets the file meta data for an image file
def get_exif(fn):
    ret = {}
    i = Image.open(fn)
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

# given a latatue calculates equatorial and polar radius of earth
def get_r_of_phi(gps):
    phi = (math.pi/180.)*gps[0]
    a = 7923.00*2640  # Equatorial radius of earth in feet
    b = 7899.86*2640  # Polar radius of earth in feet
    r = []
    r1 = a*b/math.sqrt(a*a - (a*a - b*b)*math.cos(phi))
    r.append(r1)
    r2 = r1*math.cos(phi)
    r.append(r2)
    #print 'r = ',r
    return r

# calculated the distance between two gps coordinates in feet
def calculate_distance(start_gps_deg,end_gps_deg):
   r = get_r_of_phi(start_gps_deg)
   distance_feet = []
   delta0 = (math.pi/180)*r[0]*(end_gps_deg[0] - start_gps_deg[0])
   delta1 = (math.pi/180)*r[1]*(end_gps_deg[1] - start_gps_deg[1])
   distance_feet.append(delta0)
   distance_feet.append(delta1)
   return distance_feet

#ImgDirA = 'F:/Photos_21APR2018/NorthPath/25Feet/25Feet_1' # for debugging
print '\n Browse for directory containg files from first mission: '
ImgDirA = get_directory_name()

#ImgDirB = 'F:/Photos_21APR2018/NorthPath/25Feet/25Feet_1' # for debugging
print '\n Browse for directory containg files from next mission: '
ImgDirB = get_directory_name()

#CSVDir = 'F:Photos_21APR2018/NorthPath/25Feet/Metadata' # for debugging
print '\n Browse for directory to write mission meta data:'
CSVDir = get_directory_name()
CSVfilename = raw_input('Enter CSV file name: ')
print CSVfilename
CSVfile = CSVDir + '/' + CSVfilename + '.csv'
altitude_bias = float(raw_input('Enter altitude_bias '))
k = 0
all_lines = []
this_line = [ImgDirA,ImgDirB,'min distance','H offset','W offset']
all_lines.append(this_line)
for filenames_a in os.listdir(ImgDirA):
    min_distance = 1000.0
    closest_image = 'none'
    gps_closest = [0,0]
    fn_a = ImgDirA + '/' + filenames_a
    im_a = Image.open(fn_a)
    #Image_list_a.append(im_a)
    exf_a = get_exif(fn_a)
    GPS_a = get_gps_deg(exf_a)
    #if (k == 0):
        #altitude_bias = GPS_a[2]
    height = GPS_a[2] - altitude_bias
    #print 'height ', height
    pixel_size = 0.0  
    if (height != 0):
       pixel_size = abs(4000./(2.0*height*math.tan((math.pi/180.)*FOV/2.0)))

    pixel_size = (35./26.)*pixel_size 
    #print 'pixels per foot ', pixel_size
    for filenames_b in os.listdir(ImgDirB):
        fn_b = ImgDirB + '/' + filenames_b
        im_b = Image.open(fn_b)
        #Image_list_b.append(im_b)
        exf_b = get_exif(fn_b)
        GPS_b = get_gps_deg(exf_b)
        test = calculate_distance(GPS_a,GPS_b)
        test_distance = math.sqrt(test[0]*test[0] + test[1]*test[1])
        if (test_distance < min_distance):
            min_distance = test_distance
            closest_image = filenames_b
            GPS_closest = GPS_b
        else:
            im_b.close()
    if (min_distance < 20): # only print if images closer than 20 feet
        print '*** ', filenames_a, ' *** ',closest_image, ' distance = ', min_distance
        print 'height ', height
        print 'pixels per foot ', pixel_size
        print 'GPS_a       ', GPS_a
        print 'GPS_closest ', GPS_closest
        test = calculate_distance(GPS_a,GPS_closest)
        H = int(pixel_size*test[1])
        W = int(pixel_size*test[0])
        print 'image offest in pixels H = ', H, ' W = ',W, '\n'
        this_line = [filenames_a,closest_image,min_distance,H,W]
        all_lines.append(this_line)
    im_a.close()
    k += 1

print 'Done at last! \n'
with open(CSVfile,'wb') as f_csv:
    writer = csv.writer(f_csv)
    for this_line in all_lines:
        writer.writerow(this_line)
        
    
    
    
