import PIL
from PIL import Image
from PIL.ExifTags import TAGS
import tkinter
from tkinter import *
from tkinter import filedialog
import os
#from DateTime import DateTime
#from datetime import datetime
import pandas as pd
import pickle
import numpy as np
import scipy
import scipy.linalg
import math
import csv
import cv2
#import imutils
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

root = tkinter.Tk()
FOV = 78.8 #Mavic Pro camer field of view 78.8 Deg
EARTH_MEAN_RADIUS  =   6371.01	# In km
ASTRONOMICAL_UNIT  =  149597890	# In km

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
    print (" I got %d bytes from this file." % len(data))
    return file

def save_file_name(caption):
    myFormats = [
    ('Windows Bitmap','*.bmp'),
    ('Portable Network Graphics','*.png'),
    ('JPEG / JFIF','*.jpg'),
    ('CompuServer GIF','*.gif'),
    ]
    fileName = tkinter.filedialog.asksaveasfilename(parent=root,filetypes=myFormats ,title=caption)
    if len(fileName ) > 0:
        print ("Now saving under %s" % nomFichier)
    return fileName
# gets the file meta data for an image file
def get_exif(i):
    ret = {}
    #i = PIL.Image.open(fn)
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

# find intersection between two lines
def intersect(line_P,line_Q):
    EPS = 1.0E-6
    alpha = [0,0]
    # robust routine to find the intersections between two lines
    intersects = False
    P = np.asarray(line_P)
    Mp = np.asmatrix(P).T
    #print ('Mp ',Mp)
    P0 = Mp[:,0]
    P1 = Mp[:,1]
    #print ('P0 \n',P0,'P1 \n',P1)
    Q = np.asarray(line_Q)
    Mq = np.asmatrix(Q).T
    Q0 = Mq[:,0]
    Q1 = Mq[:,1]
    #print ('Q0 \n',Q0,'\nQ1 \n',Q1) 
    A = (P1 - P0)
    B = (Q0 - Q1)
    C = (Q0 - P0)
    #print ('\n A \n',A,'\n B \n',B,'\n C \n',C)
    M = np.c_[A,B]
    #print ('M \n', M)
    try:
        Iv = np.linalg.inv(M)
        #print ('Iv \n',Iv)
        check_I = Iv*M
        #print ('check \n',check_I)
        alpha = Iv*C
        #print ('alpha \n',alpha)
        if ((alpha[0] >= -EPS and alpha[0] <= 1.+EPS) and (alpha[1] >= -EPS and alpha[1] <= 1.+EPS)):
            intersects = True
    except:
        #print('No inverse exist')
        intersects = False   
    return (intersects,alpha)

def get_contained_rectangle(corners_A,corners_B):
    """
    Given: corner points for images A and B that overlap in pixel coordinates of image A

    Find: the 4 corner points of the overlapping region (as a rectangle)

    Method: (1) Build paths around A and B
            (2) Find points from image B contained in image A and add to set of output points
            (3) Go to step (8) if total of 4 points are in set of output points
            (4) Find points from image A contained in image B and add to set of output points
            (5) Go to step (8) if total of 4 points are in set of output points
            (6) Find points of intersection between rectangles formed by corner points
                and add to set of output points.
            (7) report error if total of 4 points are in set of output points
            (8) Build bounding box form output points from (4) or (7) above.
            (9) Build and return min box
    """
    contained_points = []

    ## (2) Find points from image B contained in image A

    ##     first find a point extior to both A and B
    for i in range(4):
        if i == 0:
            x_max = max(corners_A[i][0],corners_B[i][0])
            y_max = max(corners_A[i][1],corners_B[i][1])
        else:
            x_max = max(x_max,corners_A[i][1],corners_B[i][1])
            y_max = max(y_max,corners_A[i][1],corners_B[i][1])
    x_max += 7.0*x_max
    y_max += 5.0*y_max
    #print('x_max ',x_max,'y_max ',y_max)

    ## loop on points from B with exterior point
    for i in range(0,4):
        im1 = (i+3) % 4
        #line_P = np.array(x_max,y_max,[corners_B[i][0],corners_B[i][1]])
        line_P = np.array([[x_max,y_max],[corners_B[i][0],corners_B[i][1]]])
        intersect_count = 0
        # loop on boundray of A  (edges)
        for j in range(4):
            jm1 = (j+3) % 4
            #line_Q =     np.array([[corners_A[jm1][0],corners_A[jm1][1]],[corners_A[j][0],corners_A[j][1]]])
            line_Q = np.array([[corners_A[jm1][0],corners_A[jm1][1]],[corners_A[j][0],corners_A[j][1]]])      
            (intersects,alpha) = intersect(line_P,line_Q)
            if intersects == True:
                intersect_count += 1
                #intersection = alpha[1]*(line_Q[1][:]-line_Q[0][:])+ line_Q[0][:]
                #intersections.append(intersection)
                #print(i,j,np.transpose(np.asarray(alpha)),intersects,'\n',intersection)
        if (intersect_count % 2 == 1): # an odd number of intersects implies point included
            contained_points.append([corners_B[i][0],corners_B[i][1],1.])
                
    n_contained_points = len(contained_points)
    #print ('\n n_contained_points ',n_contained_points)

    if (n_contained_points < 4 ): # Find points from image A contained in image B
        ## loop on points from A with exterior point
        for i in range(0,4):
            im1 = (i+3) % 4
            #line_P = np.array(x_max,y_max,corners_A[i][0],corners_A[i][1])
            line_P = np.array([[x_max,y_max],[corners_A[i][0],corners_A[i][1]]])      
            intersect_count = 0
            # loop on boundray of B  (edges)
            for j in range(4):
                jm1 = (j+3) % 4
                #line_Q =     np.array([[corners_B[jm1][0],corners_B[jm1][1]],[corners_B[j][0],corners_B[j][1]]])
                line_Q = np.array([[corners_B[jm1][0],corners_B[jm1][1]],[corners_B[j][0],corners_B[j][1]]])
                (intersects,alpha) = intersect(line_P,line_Q)
                if intersects == True:
                    intersect_count += 1
                    intersection = alpha[1]*(line_Q[1][:]-line_Q[0][:])+ line_Q[0][:]
                    #intersections.append(intersection)
                    #print(i,j,np.transpose(np.asarray(alpha)),intersects,'\n',intersection)
            if (intersect_count % 2 == 1): # an odd number of intersects implies point include
                contained_points.append([corners_A[i][0],corners_A[i][1],1.])
                #print (i,j,'intersect_count ',intersect_count)
    n_contained_points = len(contained_points)
    #print ('\n n_contained_points ',n_contained_points)

    if (n_contained_points < 4 ): # intersections of edges from image A and image B
       ## loop boundray of A (edges)
        for i in range(0,4):
            im1 = (i+3) % 4
            #line_P = np.array(x_max,y_max,[corners_A[i][0],corners_A[i][1]]])
            line_P = np.array([[corners_A[im1][0],corners_A[im1][1]],[corners_A[i][0],corners_A[i][1]]])

            # loop on boundray of B  (edges)
            for j in range(4):
                jm1 = (j+3) % 4
                #line_Q =     np.array([[corners_B[jm1][0],corners_B[jm1][1]],[corners_B[j][0],corners_B[j][1]]])
                line_Q = np.array([[corners_B[jm1][0],corners_B[jm1][1]],[corners_B[j][0],corners_B[j][1]]])
                (intersects,alpha) = intersect(line_P,line_Q)
                if intersects == True:
                    x_0 = alpha[0]*line_P[1][0] + (1-alpha[0])*line_P[0][0]
                    y_0 = alpha[0]*line_P[1][1] + (1-alpha[0])*line_P[0][1]
                    contained_point = np.zeros(3)
                    contained_point[0]  = x_0
                    contained_point[1]  = y_0
                    contained_point[2]  = 1.0
                    #print(i,j,np.transpose(np.asarray(alpha)),intersects,'\n',contained_point)
                    contained_points.append(contained_point)

    n_contained_points = len(contained_points)
    contained_points = np.matrix(contained_points)
    #print ('\n n_contained_points ',n_contained_points)
    #print ('containes_points: \n',contained_points)
    min_box = get_min_box(contained_points)
    return min_box

def get_min_box(pts):
    """
    Giben: a set (x,y) of boundary points
    Find: the min box that can contain only points in set
    Method: (1) build bounding box
            (2) find point closest to each corner of bb
            (3) corners of min box from corner closest points
    """
    n_pts = len(pts)
    #print ('n_pts ',n_pts,' pts \n',pts)
    # build bounding box
    for i in range(n_pts):
        if i == 0:
            x_min = pts[i,0]
            x_max = pts[i,0]
            y_min = pts[i,1]
            y_max = pts[i,1]
        else:
            x_min = min(x_min,pts[i,0])
            x_max = max(x_max,pts[i,0])
            y_min = min(y_min,pts[i,1])
            y_max = max(y_max,pts[i,0])
    # find points closest to each corner of bb
    for i in range(n_pts):
        test_Q1 = math.sqrt((pts[i,0]-x_max)*(pts[i,0]-x_max)+(pts[i,1]-y_min)*(pts[i,1]-y_min))
        test_Q2 = math.sqrt((pts[i,0]-x_min)*(pts[i,0]-x_min)+(pts[i,1]-y_min)*(pts[i,1]-y_min))
        test_Q3 = math.sqrt((pts[i,0]-x_min)*(pts[i,0]-x_min)+(pts[i,1]-y_max)*(pts[i,1]-y_max))
        test_Q4 = math.sqrt((pts[i,0]-x_max)*(pts[i,0]-x_max)+(pts[i,1]-y_max)*(pts[i,1]-y_max))
        if i == 0: # initialize min distances
            i_Q1 = i
            i_Q2 = i
            i_Q3 = i
            i_Q4 = i
            min_Q1 = test_Q1
            min_Q2 = test_Q2
            min_Q3 = test_Q3
            min_Q4 = test_Q4
        else:
            if (test_Q1 < min_Q1):
                i_Q1 = i
                min_Q1 = test_Q1
            if (test_Q2 < min_Q2):
                i_Q2 = i
                min_Q2 = test_Q2
            if (test_Q3 < min_Q3):
                i_Q3 = i
                min_Q3 = test_Q3
            if (test_Q4 < min_Q4):
                i_Q4 = i
                min_Q4 = test_Q4
    top = min(pts[i_Q1,1],pts[i_Q2,1])
    bottom = max(pts[i_Q3,1],pts[i_Q4,1])
    left = max(pts[i_Q2,0],pts[i_Q3,0])
    right = min(pts[i_Q1,0],pts[i_Q4,0])

    min_box = [[left,top,1],
               [left,bottom,1.0],
               [right,bottom,1.0],
               [right,top,1.0]]
              
    min_box = np.matrix(min_box)
    #print ('min_box \n',min_box)
    return min_box
               
def plot_response_surface(X,Y,Z,X_gd,Y_gd,Z_gd,pf):

    # Plot response surface
    fig = plt.figure()
    #fig.gca(
    ax = plt.axes(projection='3d')
    #collect data
##    
##    X_grid = np.array(X)
##    Y_grid = np.array(Y)
##    X,Y = np.meshgrid(X_grid,Y_grid)
##    

    # Plot the surface
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
                           linewidth=1,antialiased=False)
    
    ax.plot3D(X_gd,Y_gd,Z_gd)
    ax.scatter3D(X_gd,Y_gd,Z_gd,  cmap='Greens');
    # Customize the z axis.
    ax.set_zlim(0.6, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
              

    plt.show()
        

    # use pickle file passed to routine
    with open(pf, 'wb') as fid:
        pickle.dump(ax, fid) # save as pickle file for later reuse
    
    plt.pause(1)
    x = input('Enter any key to continue: ')
    plt.close()
    
    return

def plot_solution(ImageA,ImageB,croppedA,croppedB):
    #plt.ion()
    plt.subplot(2,2,1),plt.imshow(ImageA),plt.title('Image A')
    #plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(ImageB),plt.title('Image B')
    #plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(croppedA),plt.xlabel('A cropped')
    #plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(croppedB),plt.xlabel('B cropped')
    #plt.xticks([]), plt.yticks([])
    plt.show()
    plt.pause(1)
    x = input('Enter any key to continue: ')
    plt.close()
    return

def cal_covariance(test_image):
    imA =test_image.convert('RGB')
    (W,H) = imA.size
    #print('test image shape',imA.size)
    imA = np.array(imA)
    #imA = np.transpose(imA)
    augmented_vector = []
    
    #print('W',W,'H',H)
    for i in range (0 , H):
        for j in range (0,W):
            k = W*i + j - 1
            (Rk,Gk,Bk) = imA[i,j]
            augmented_vector.append([i,j,Rk,Gk,Bk])
    cov = np.cov(np.transpose(augmented_vector))
    return cov

def calculate_sse(A,B):
    D = A-B
    SSE = np.sum(np.dot(D,np.transpose(D)))
    print (' SSE_loss ',SSE)
    return SSE

def calculate_eig_loss(A,B):
    (lamda,eig_vec) = scipy.linalg.eig(A,B)
    real_lamda = scipy.real(lamda)
    #print('lamda ',real_lamda)
    #print('eig_vec',eig_vec)
    eig_loss = 0.
    for i in range (0,len(real_lamda)):
        eig_loss += (np.log(real_lamda[i]))*(np.log(real_lamda[i]))                                         
    eig_loss = math.sqrt(eig_loss)
    #print (' eigenvalue_loss',eig_loss)
    return eig_loss

        
def calculate_gradient(fnc,M):
    M1 = M
    EPS = 1.0E-9
    FACTOR = 5
    f_0 = fnc(M)
    #print('f_0',f_0)
    #print('M:\n',M)
    #print('len(M)',len(M))
    grad = np.zeros([len(M),len(M)],dtype = float)
    #print('grad',grad)
    M_star = M1
    for i in range (len(M)-1):
        for j in range (2,len(M)):
            del_x = FACTOR
            M_star[i][j] += del_x
            del_y = fnc(M_star)-f_0

            #print('i',i,'j',j,'del_x',del_x,'del_y',del_y)
            #print ('M_star[',i,'][',j,'] ',M_star[i][j])
            M_star = M1
            try:
                grad[i][j] = del_y/del_x
            except:
                grad[i][j] = EPS
    #print('grad: \n',grad)
    return grad

class Proximity_stitcher:
    def __init__(self):
        self.ImageA = None
        self.ImageB = None
        self.fn_pickle = ""
        self.cv2_ImageA = None
        self.cv2_ImageB = None
        self.Ma = np.identity(3,dtype=np.float64)
        self.Mb = np.identity(3,dtype=np.float64)
        altitude_bias_left_image = 0.
        self.exf_A = []
        self.GPS_A = []
        altitude_bias_right_image = 0.
        self.exf_B = []
        self.GPS_B = []
        self.test = []
        self.pixel_sizeA = 0.0
        self.h_imageA = 0
        self.w_imageA = 0
        self.pixel_sizeB = 0.0
        self.h_imageB = 0
        self.w_imageB = 0
        self.epochs = 0
        self.bb_A = []
        self.bb_B = []
        self.cov_A = []
        self.cov_B = []
        self.A_cropped = None
        self.B_cropped = None

    def proximity_stitch(self, imA,imB,fn_pickle):
        self.ImageA = imA
        self.altitude_bias_left_image = float(input('Enter altitude_bias for left image: '))

        # get gps data from left image
        self.exf_A = get_exif(self.ImageA)
        self.GPS_A = get_gps_deg(self.exf_A)
        self.height_left_image = self.GPS_A[2] - self.altitude_bias_left_image
            
        self.ImageB = imB
        self.altitude_bias_right_image = float(input('Enter altitude_bias for right image: '))

        self.fn_pickle = fn_pickle   #Pickle file name
        
        # get gps data from right image
        self.exf_B = get_exif(self.ImageB)
        self.GPS_B = get_gps_deg(self.exf_B)
        self.height_right_image = self.GPS_B[2] - self.altitude_bias_right_image

        # calculate distance between images from GPS data
        self.test = calculate_distance(self.GPS_A,self.GPS_B) # Distance camera position for Imsage A to image B (feet)
        print('offset distance between images = ',self.test, ' feet')

        # calculate pixel size (left image)
        self.ImageA_width = 2.0*self.height_left_image*math.tan((math.pi/180.)*FOV/2.0)
        self.ImageA_width = (26./35.)*self.ImageA_width
        (self.w_imageA, self.h_imageA)   = self.ImageA.size
        
        if (self.ImageA_width != 0.):
           self.pixel_sizeA = abs(self.ImageA_width/float(max(self.w_imageA,self.h_imageA)))
        print ('pixels size A (feet) ', self.pixel_sizeA)

        # calculate pixel size (right image)
        self.ImageB_width = 2.0*self.height_left_image*math.tan((math.pi/180.)*FOV/2.0)
        self.ImageB_width = (26./35.)*self.ImageB_width
        (self.w_imageB, self.h_imageB)   = self.ImageB.size
        
        if (self.ImageB_width != 0.):
           self.pixel_sizeB = abs(self.ImageB_width/float(max(self.w_imageB,self.h_imageB)))
        print ('pixels size B(feet) ', self.pixel_sizeB)

        #Ma is idenity matrix but could include GPS error for Image A
        self.Ma = np.identity(3,dtype=np.float64)
        #print ('Ma \n',self.Ma)

        # Build translation matrix Mb (first estimate)
        self.Mb = np.identity(3,dtype=np.float64)
        try:
            #self.Mb[0][2] = self.test[1]/self.pixel_sizeB  # + 20
            self.Mb[0][2] = 2580
            #self.Mb[1][2] = -self.test[0]/self.pixel_sizeB # - 645      # for debugging purposes added 10
            self.Mb[1][2] = -346
        except:
            self.Mb[0][2] = 0.
            self.Mb[1][2] = 0.
        print ('Mb \n',self.Mb)

        # Brute force metnod
        (X_grid,Y_grid,self.e_losses) = self.find_best_match_brute_force(10)
        plot_size = int(len(X_grid))
        #print('lenght of e_losses',plot_size,'\n e_losses: \n',self.e_losses)
        X = np.array(X_grid)
        Y = np.array(Y_grid)
        X, Y = np.meshgrid(X, Y)
        self.e_losses = np.array(self.e_losses)
        Z = self.e_losses.reshape((plot_size,plot_size))
        print('lenght of e_losses',plot_size,'\n e_losses: \n',Z)
        plot_response_surface(X,Y,Z)

     
##      Reset Mb
        self.Mb = np.identity(3,dtype=np.float64)
        try:
            self.Mb[0][2] = self.test[1]/self.pixel_sizeB  # + 20
            self.Mb[1][2] = -self.test[0]/self.pixel_sizeB # - 645      # for debugging purposes added 10
        except:
            self.Mb[0][2] = 0.
            self.Mb[1][2] = 0.
        print ('Reset Mb \n',self.Mb)
##        # Loop on epochs
        epochs = int(input('Enter maximum number of epochs: '))
        (X_gradent_descent,Y_gradent_descent,self.e_losses) = self.find_best_match(epochs) 

##        X_gd = np.array(X_gradent_descent)
##        Y_gd = np.array(Y_gradent_descent)
##        #X, Y = np.meshgrid(X, Y)
##        Z_gd = np.array(self.e_losses)
##        #Z = self.e_losses.reshape((plot_size,plot_size))
##        plot_response_surface(X,Y,Z,X_gd,Y_gd,Z_gd,self.fn_pickle)
        
        return (self.A_cropped,self.B_cropped)

    def e_loss_from_M(self,Mb):
        self.Mb = Mb
        e_loss = 0.
        (self.A_cropped,self.B_cropped) = self.calculate_overlaps_with_matrix(Mb)
        self.cov_A = cal_covariance(self.A_cropped)
        self.cov_B = cal_covariance(self.B_cropped)
        e_loss = calculate_eig_loss(self.cov_A,self.cov_B)
        return e_loss

    # calculate argminMb(e_loss)
    def find_best_match(self,epochs):
        gamma = 10
        X = []
        Y = []
        e_loss_min = 1.0E+6
        e_loss_cutoff = 0.1
        e_loss_last = e_loss_min
        e_losses = []
        self.Mb_last = self.Mb
        e_loss = self.e_loss_from_M(self.Mb)
        e_losses.append(e_loss)
        for epoch in range(epochs):            
            print('e_loss',e_loss)
            if e_loss < e_loss_min:
                e_loss_min = e_loss            
            X.append(self.Mb[0][2])
            Y.append(self.Mb[1][2])         
            if e_loss_last < 0.9*e_loss:
                print('reversal in e_loss',e_loss_last,e_loss)
                break
            e_loss_last = e_loss
            if e_loss_min < e_loss_cutoff:
                print('e_loss_min',e_loss_min)
                break
            grad = calculate_gradient(self.e_loss_from_M,self.Mb)
            print('grad \n',grad)
            gamma = 10
            
##            if epoch != 0:
##                grad_last = calculate_gradient(self.e_loss_from_M,self.Mb_last)
##                print('self.Mb_last \n',self.Mb_last)
##                print('grad_last \n',grad_last)
##                denominator = np.linalg.norm(grad-grad_last)**2
##                print ('denominator',denominator)
##                Mb_Mb_last = self.Mb-self.Mb_last
##                print ('Mb_Mb_last \n',Mb_Mb_last )
##                print ('dot product \n',np.transpose(Mb_Mb_last)*(grad-grad_last) )
##                norm = np.linalg.norm(np.transpose(Mb_Mb_last)*(grad-grad_last))
##                print ('norm',norm)
##                print ('(grad-grad_last)',(grad-grad_last))
##                try:
##                    gamma = norm/denominator
##                    print('%%%%%% gamma %%%%%%',gamma)
##                except:
##                    print('##### exception on gamma calculation #####')
##                    gamma = 10

            try:            
                print('Before update self.Mb[',epoch,'] \n',self.Mb)
                self.Mb = self.Mb - gamma*grad
                print('After update self.Mb[',epoch,'] \n',self.Mb)
            except:
                print('****** Exception ****** self.Mb[',epoch,'] \n',self.Mb)
            e_loss = self.e_loss_from_M(self.Mb)
            e_losses.append(e_loss)
        return (X,Y,e_losses)

    def find_best_match_brute_force(self,tile_size):
        e_loss_min = 100.
        e_losses = []
        X = []
        Y = []
        Mb_working = self.Mb
        dx_start = Mb_working[0][2] - 10*tile_size
        dy_start = Mb_working[1][2] - 10*tile_size
        dx = dx_start
        dy = dy_start

        for i in range(40):
            dx = dx_start
            Mb_working[0][2] = dx
            for j in range(40):
                e_loss = self.e_loss_from_M(Mb_working)
                if i == 0:
                    X.append(dx)
                if e_loss < e_loss_min:
                    e_loss_min = e_loss
                    self.Mb = Mb_working

                e_losses.append(e_loss)
                dx = dx + tile_size
                #print('****Mb_working[0][2]****',Mb_working[0][2])
                Mb_working[0][2] = dx
            
            Y.append(dy)
            dy = dy + tile_size
            Mb_working[1][2] = dy
        print('&*&* Best e_loss &*&*',e_loss_min)
        print('&*&* Best Mb &*&* \n ',self.Mb)
        return (X,Y,e_losses)
                              
        

    def calculate_overlaps_with_matrix(self,Mb):
        self.Mb = Mb


        """
        Given: 2 images A and B of pixel size pixel_sizeA and
        pixel sizeB that overlap and are related my a matrix M i.e. B = M*A
        (A = M.inverse*B)

        Find: the overlap of the two images(in local pixel coordinates of both)
        and return the cropped regions as new images

        Method: (1) Find corner points of each image in coordinates of image A
                (2) Build paths around A and B
                (3) Find intersection of each edge of B with each edge of A
                (4) Divide each edge at intersection points to form new line segments
                (5) Collect line segments that form a closed path contained in both paths A and B
                (6) Build bounding box form closed path from(5) above
                (7) Crop image A to bb
                (8) Translate bb back to B image reference  and Crop Image B
                (9) Return cropped images
        
        """
        # set defaults to entire images
        overlap_AtoB = self.ImageA
        overlap_BtoA = self.ImageB
        
        # find corner points
        (w_imageA, h_imageA)   = self.ImageA.size
        corners_A = [[w_imageA,0,1],[0,0,1],[0,h_imageA,1],[w_imageA,h_imageA,1]]
        corners_A_star = np.transpose(np.matmul(self.Ma,np.transpose(corners_A)))
        ##print('corners_A_star\n',corners_A_star)
    

        (w_imageB, h_imageB)   = self.ImageB.size
        #print ('w_imageB ' , w_imageB, ' h_imageB ', h_imageB)
        corners_B = [[w_imageB,0,1],[0,0,1],[0,h_imageB,1],[w_imageB,h_imageB,1]]

        # Map image B corners to image A coordinates
        #print ('self.Mb',self.Mb)
        corners_B_star = np.transpose(np.matmul(self.Mb,np.transpose(corners_B)))
        # added scale factor 1/w JHC 6/23/2019
        try:
            corners_B_star *= 1.0/self.Mb[2][2]
            #print('Worked!!')
            #print('corners_B_star\n',corners_B_star)
        except:
            #print("Didn't Worked!!")
            corners_B_star = corners_B_star  # in case self.Mb[2][2 ] = 0 
        #print ('corners_B_star \n',corners_B_star)
        
        # find contained points
        min_box = get_contained_rectangle(corners_A,corners_B_star)    
        self.bb_A = [max(0,int(min_box[0,0])),max(0,int(min_box[0,1])), int(min_box[2,0]),int(min_box[2,1])]
        #print ('min_box \n',min_box,'\n self.bb_A \n',self.bb_A)
        #overlap_AtoB = self.imcrop(self.ImageA, self.bb_A)
        overlap_AtoB = self.ImageA.crop(self.bb_A)

        Mb_inverse = np.linalg.inv(Mb)
        Mb_inverse[0,2] = Mb_inverse[0,2]*self.pixel_sizeB/self.pixel_sizeA
        Mb_inverse[1,2] = Mb_inverse[1,2]*self.pixel_sizeB/self.pixel_sizeA
        #print('Mb_inverse \n',Mb_inverse)
              

        #min_box_inv = Mb_inverse*min_box.T
        min_box_inv = np.transpose(np.matmul(Mb_inverse,np.transpose(min_box)))

        self.bb_B = [max(0,int(min_box_inv[0,0])),max(0,int(min_box_inv[0,1])), int(min_box_inv[2,0]),int(min_box_inv[2,1])]
        #print ('min_box_inv \n',min_box_inv,'\n self.bb_B \n',self.bb_B)

        #overlap_BtoA = self.imcrop(self.ImageB, self.bb_B)
        overlap_BtoA = self.ImageB.crop(self.bb_B)
        return [overlap_AtoB,overlap_BtoA]



def stitch_images():
    # load the two images and resize them to have a width of 400 pixels

    # Get first file to process
    print ('Please Browse for first image to be stitched')
    ImageFile_A = get_file_name('Select first (left) image file')
    img1 = ImageFile_A.name
    ImageA = PIL.Image.open(img1)

    # Get second file to process
    print ('Please Browse for second image to be stitched')
    ImageFile_B = get_file_name('Select second (right) image file')  
    img2 = ImageFile_B.name
    ImageB = PIL.Image.open(img2)

    # Get directory name to hold pickle file
    print ('Please select directory to hold pickle file: ')
    pickle_file_directory = get_directory_name('Please select directory to hold pickle file: ')
    fn_pickle = pickle_file_directory  + 'respinse_surface_plot.pickle'
    print ('pickle file ',fn_pickle)


    

    stitcher = Proximity_stitcher()
    (imgA_cropped, imgB_cropped) = stitcher.proximity_stitch(ImageA,ImageB,fn_pickle)
    
    sse =  calculate_sse(stitcher.cov_A,stitcher.cov_B)

    # Plot result
    # show the images

    # Do you want to show input images?
    Yes_No_string = str(input ('Do you want to show solution (y/n) '))
    if Yes_No_string == 'y' or Yes_No_string == 'Y':
        plot_solution(ImageA,ImageB,imgA_cropped, imgB_cropped)

    save_plots = input('do you want to save cropped images (y/n)? ')   
    if (save_plots == 'y' or save_plots == 'Y'):

        A_x_offset = stitcher.bb_A[0]
        #print ('A_x_offset',A_x_offset)

        fna_cropped = ImageFile_A.name[0:len(ImageFile_A.name)-4]+ '_cropped' +'.JPG'
        ImageA_cropped = ImageA.crop(stitcher.bb_A)
        ImageA_cropped.save(fna_cropped)

        B_x_offset = stitcher.bb_B[0]
        #print ('B_x_offset',B_x_offset)
        
        fnb_cropped = ImageFile_B.name[0:len(ImageFile_B.name)-4]+ '_cropped' +'.JPG'
        ImageB_cropped = ImageB.crop(stitcher.bb_B)
        ImageB_cropped.save(fnb_cropped)

def main():
    stitch_images()
    return

main()
"""
if __name__ == "main":
     # execute only if run as a script
     main()
"""      
    
