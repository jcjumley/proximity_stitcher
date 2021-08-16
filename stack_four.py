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
FOV = 78.9 #Mavic Pro camera field of view 78.9 Deg
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
# given i,j and J calculate k
def calculate_k_from_i_and_j(i,j,J):
    if (i%2 == 0):
        k = (i+1)*J - j
    else:
        k = i*J + j + 1
    return k

# given k and J calculate i,j
def calculate_i_and_j_from_k(k,J):
    i = (k-1)/J
    if ((i%2) == 0):
        j = J - (k - J*i)
    else:
        j = k - J*i - 1
    return[i,j]

#ImgDirA = 'F:/Photos_21APR2018/NorthPath/25Feet/25Feet_1' # for debugging
print '\n Browse for directory containg files to stack: '
ImgDirA = get_directory_name()
print ImgDirA

#CSVDir = 'F:Photos_21APR2018/NorthPath/25Feet/Metadata' # for debugging
print '\n Browse for directory to write mission meta data:'
CSVDir = get_directory_name()
CSVfilename = raw_input('Enter CSV file name: ')
print CSVfilename

CSVfile = CSVDir + '/' + CSVfilename + '.csv'
altitude_bias = float(raw_input('Enter altitude_bias: '))
#J = input('Enter number of waypoints per row: ')
J = input('Enter number of waypoints per row: ')
all_lines = []
H = []
W = []
#overlapping_file = []
for i in range(0,4):
   H.append(0)
   W.append(0)
   #overlapping_file.append('NONE') 

this_line = [ImgDirA]
all_lines.append(this_line)
this_line = ['File_a','File_b','W offset_b','H offset_b','File_c','W offset_c','H offset_c','File_d','W offset_d','H offset_d']
all_lines.append(this_line)
any_overlap = False
filenames = []
offset_file = []
Image_count = 0

for filename_a in os.listdir(ImgDirA):
    filenames.append(filename_a)
    Image_count += 1
           
offset_file.append('NONE')
offset_file.append('NONE')
offset_file.append('NONE')
offset_file.append('NONE')                  
for i in range(0,Image_count/J,2):
    for j in range(0,J,2):
        ks = []
        k = calculate_k_from_i_and_j(i,j,J)  # step over 1st waypoint
        offset_file[0] = filenames[k]
        ks.append(k)
        #print 'k', k
        next_j = j + 1
        if (next_j > J):
            next_j = J
             
        next_i = i + 1
        if (next_i > Image_count/J):
            next_i = Image_count/J

        next_k = calculate_k_from_i_and_j(i,next_j,J) # step over 1st waypoint
        #print 'next_k', next_k
        ks.append(next_k)
        next_k = calculate_k_from_i_and_j(next_i,j,J) # step over 1st waypoint 
        #print 'next_k', next_k
        ks.append(next_k)
        next_k = calculate_k_from_i_and_j(next_i,next_j,J) # step over 1st waypoint
        #print 'next_k', next_k
        ks.append(next_k)

        any_overlap = False
        fn_a = ImgDirA + '/' + filenames[k]
        im_a = Image.open(fn_a)
        exf_a = get_exif(fn_a)
        GPS_a = get_gps_deg(exf_a)
        height = GPS_a[2] - altitude_bias
        #print 'height ', height
        pixel_size = 0.0
        if(height != 0):
            pixel_size = abs(4000./(2.0*height*math.tan((math.pi/180.)*FOV/2.0)))
            pixel_size = (35./26.)*pixel_size

        for neigbor in range(1,4):
            fn_b = ImgDirA + '/' + filenames[ks[neigbor]]
            im_b = Image.open(fn_b)
            exf_b = get_exif(fn_b)
            GPS_b = get_gps_deg(exf_b)
            test = calculate_distance(GPS_a,GPS_b)
            test_distance = math.sqrt(test[0]*test[0]+test[1]*test[1])
            pixel_offset_W = int(pixel_size*test[1])
            pixel_offset_H = int(pixel_size*test[0])
            overlaps = False
            if ((abs(pixel_offset_W) <= 4000) & (abs(pixel_offset_H) <= 3000)):
                any_overlap = True
                overlaps = True

            offset_file[neigbor] = filenames[ks[neigbor]]
            W[neigbor] = pixel_offset_W
            H[neigbor] = pixel_offset_H

        print 'Base Image GPS ', GPS_a,'height',height,'pixels per foot ', pixel_size
        print '*** ', offset_file[0], ' *** ', offset_file[1], '*** ',offset_file[2],'***',offset_file[3]
        print '   offsets in pixels     ',ks[1],W[1],H[1],'      ',ks[2],W[2],H[2],'      ',ks[3],W[3],H[3]
        
        this_line = [offset_file[0],offset_file[1],W[1],H[1],offset_file[2],W[2],H[2],offset_file[3],W[3],H[3]]
        all_lines.append(this_line)
    im_a.close()

print 'Done at last! \n'
with open(CSVfile,'wb') as f_csv:
    writer = csv.writer(f_csv)
    for this_line in all_lines:
        writer.writerow(this_line)
        
    
    
    
