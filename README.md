# proximity_stitcher
proximity_stitcher is a project to create a python program for stitch together two images that are known to overlap from the GPS data they contain. 
The objective is to obtain the best match while using the smallest possible overlap when collecting the images.
For this project Images were collected using a DJI mavic pro with a 12 Mpixel camera with a FOV of 78.8 deg and a crop factor of 26/35 (f/2.2). Also, the pixel size was calculated from the altitude of the drone recovered from the image GPS data. To calculate the "true" height above ground level it is necessary to subtract the altitude bias.
The program prompts for all required inputs: it is only necessary to know the file name and path of the images and the altitude bias.
As a point of reference I have also included a feature based stitching program from Adrin at PyImageSearch. BUT note to run ths program you will need to have SURF and SIFT as part of your python installation (not standard).
