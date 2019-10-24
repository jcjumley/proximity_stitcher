# proximity_stitcher
proximity_stitcher is a project to create a python program for stitch together two images that contain GPS data. The objective is to obtain the best possible match while using the smalest possible overlap.
For this project Images were collected withy a DJI mavic pro with a 12 mpixel camera with a FOV of 78 deg and a crop factor of 26/35.  Also, the pixel size was calculated from the altitude od the drone recovered from the image GPS data. To calculate the "true" height above ground level it is necessary to subtact the altitude bias.
