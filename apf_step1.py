###############################################################################
#                                                                             #
#                   The Orbits of Long Period Exoplanets:                     #
#                                  GRAND:                                     #
#            Gaussian-Modeled Relative Astrometry for NIRC2 Data              #
#                                                                             #
#                        Written by Logan A. Pearce (2018)                    #
#                                                                             #
###############################################################################
#
# GRAND is a method for determining the position of a secondary object relative to a primary it is orbiting
# in NIRC2 images from Keck II.  GRAND models the primary and secondary PSF as the sum of two 2-D Gaussians -
# a narrow Gaussian modeling the diffraction limited core and a wider Gaussian capturing the Airy rings.  It
# uses a Gibbs Sampler Metroplis-Hasting Markov Chain Monte Carlo algorithm to optimize the parameters of model
# fit to image data.  It converts the object positions in pixel space in the image to separation and position
# angle, then corrects for NIRC2 distortion, orientation correction, atmospheric and chromatic corrections.
#
################################################################################
#          Step 1: Generate an intial guess for the model fit.                 #
################################################################################
# In step 1, the script takes in an image file, asks the user to identify the location of the two objects
# and an empty background area, and generates an initial guess for the locations and estimate of the background
# levels.
#
# Inputs:
#  - Folder of processed NIRC2 image files
# 
# Output:
#  - A file title "filename_intialguess" in the same directory as the image file containing
#      a x and y position for star, companion, and an estimate of the sky background levels.  This file is
#      imported into step 2.
#
# From the terminal, execute as follows:
#   python grand_step1.py path_to_images_folder
#   example: python grand_step1.py /Users/loganpearce/Desktop/UTexas/Astro_research_data/data/GSC6214/2018_07_01
#
# If running on TACC, log in to TACC using the -XY ssh flag to allow user interaction with scripts via ssh, and
# execute step 1 in an idev session using the above syntax.

import matplotlib
matplotlib.use('TkAgg')
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import argparse

############################################
## Make a list of image files in the folder:

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="path from this script to directory containing image files to analyze, without \
    concluding '/' ", type=str)
args = parser.parse_args()
directory=args.directory

#os.system('ls '+directory+'/*.LDFBC.fits > list')
os.system('ls '+directory+'/*.fits > list')

############################################
# Find the pixel with the max value within the aperture centered at the orginal guess
def findmax(data):  
    from numpy import unravel_index
    m = np.argmax(data)
    c = unravel_index(m, data.shape)
    return c

with open('list') as f:
    z = f.read().splitlines()

for line in z:
    print 'Loading image ',line
    image1 = fits.open(line, ignore_missing_end=True)
    image = image1[0].data
   ############### Click on the image to get initial guess of center of companion and star ##################
    print 'Press "D" key to select center of companion'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',vmin=np.percentile(image,5),vmax=np.percentile(image,97))
    ax.set_title('Click center of companion')
    coordc = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)
        global coordc
        coordc.append((ix, iy))
        if len(coordc) == 1:
            fig.canvas.mpl_disconnect(cid)
        plt.close()
        return coordc
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    print coordc
    coordc=coordc[0]
    xmc,ymc = coordc[0],coordc[1]
    xmc,ymc = int(xmc),int(ymc)

    print 'Press "D" key to select center of star'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower')
    ax.set_title('Click center of star')
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)

        global coords
        coords.append((ix, iy))
        if len(coords) == 1:
            fig.canvas.mpl_disconnect(cid)
        plt.close()
        return coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    print coords
    coords=coords[0]
    xms,yms = coords[0],coords[1]
    xms,yms = int(xms),int(yms)

    print 'Press "D" key to select backgroung sample'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',vmin=np.percentile(image,5),vmax=np.percentile(image,95))
    ax.set_title('Click an empty sky area')
    coordc = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(ix, iy)

        global coordc
        coordc.append((ix, iy))
        if len(coordc) == 1:
            fig.canvas.mpl_disconnect(cid)
        plt.close()
        return coordc
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    print coordc
    coordc=coordc[0]
    bkgdx,bkgdy = coordc[0],coordc[1]
    bkgdx,bkgdy = int(bkgdx),int(bkgdy)

    #### Make an aperture around the click location and find the pixel with the max flux within that aperture #####
    ymins=yms-11
    ymaxs=ymins+21
    xmins=xms-11
    xmaxs=xmins+21
    aprs = image[ymins:ymaxs,xmins:xmaxs]
    #Find max pixel within aperture and call that pixel the initial guess:
    cs = findmax(aprs) #[0]=Y,[1]=X
    xcs,ycs = xmins+cs[1]+0.5,ymins+cs[0]+0.5
    print 'Initial guess for star location:',xcs,ycs

    # Initial guess of companion location of max:
    yminc=ymc-11
    ymaxc=yminc+21
    xminc=xmc-11
    xmaxc=xminc+21
    aprc = image[yminc:ymaxc,xminc:xmaxc]
    cc = findmax(aprc)
    xcc,ycc = xminc+cc[1]+0.5,yminc+cc[0]+0.5
    print 'Initial guess for companion location:',xcc,ycc

    # Write out initial guess to file:
    #newfile = directory+'/'+line.split('.')[-1]+ '_initialguess'
    newline = line.split('/')[-1]
    newfile = directory+'/'+newline.split('.')[2]+ '_initialguess'
    os.system('touch '+newfile)
    print newfile
    #newfile = line.split('.')[2]+ '_initialguess'
    string = str(xcs)+' '+str(ycs)+' '+str(xcc)+' '+str(ycc)+' '+str(bkgdx)+' '+str(bkgdy)
    k = open(newfile, 'w')
    k.write(string + "\n")
    k.close()

os.system('rm list')
