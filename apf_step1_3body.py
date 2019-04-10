'''
############################ LAPF: Logan's Analytical PSF Fitter ##############################
                                          Step 1
                               written by Logan Pearce, 2019
###############################################################################################
    Fits a Gaussian 2d PSF model to NIRC2 data for companion to a central (unobscured) star
    using a Gibbs sampler Metroplis-Hastings MCMC.  Runs in parallel, where each process acts an 
    independent walker.  For details, see Pearce et. al. 2019.
       Step 1: Locate central object, companion, and empty sky area in the image
       Step 2: MCMC iterates on parameters of 2D model until a minimum number of trials
               are conducted on each parameter.  Each process outputs their chain to an 
               individual file
       Step 3: Take in output of step 2 and apply corrections to determine relative separation
               and position angle and corresponding metrics.

# Requires:
#   python packages astropy, numpy, mpi4py
#
# Input:
#   Step 1: Image (fits file)
#   Step 2: Image (fits file), output from step 1 (text file)
#   Step 3: Output from step 2 (.csv files from each process)
#
# Output:
#   Step 1: Text file of (x,y) location for object 1, object 2, object 3, background area
#
# usage (local): python apf_step1.py path_to_images_folder
    example: python apf_step1_3body.py ../IC382-25/2013/

# User defined settings:
#    none

'''
 
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import argparse
from matplotlib.colors import LogNorm

############################################
## Make a list of image files in the folder:

parser = argparse.ArgumentParser()
parser.add_argument("folder",type=str)
args = parser.parse_args()
directory=args.folder

os.system('ls '+directory+'*.LDIF.fits > list')

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
    image1 = fits.open(line)
    image = image1[0].data
   ############### Click on the image to get initial guess of center of companion and star ##################
    print 'Press "D" key to select center of A'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',norm=LogNorm())
    ax.set_title('Hover mouse over A and type D')
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
    xma,yma = coordc[0],coordc[1]
    xma,yma = int(xma),int(yma)

    print 'Click to select center of B'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',norm=LogNorm())
    ax.set_title('Hover mouse over B and click')
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
    xmb,ymb = coords[0],coords[1]
    xmb,ymb = int(xmb),int(ymb)

    print 'Press "D" key to select center of B'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',norm=LogNorm(),interpolation='bilinear')
    ax.set_title('Hover mouse over C click')
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
    xmc,ymc = coords[0],coords[1]
    xmc,ymc = int(xmc),int(ymc)

    print 'Press "D" key to select backgroung sample'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',vmin=np.percentile(image,5),vmax=np.percentile(image,95))
    ax.set_title('Hover mouse over an empty sky area and click')
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
    ymins,ymaxs=ycb-4,ycb+4
    xmins,xmaxs=xcb-4,xcb+4
    aprs = image[ymins:ymaxs,xmins:xmaxs]
    #Find max pixel within aperture and call that pixel the initial guess:
    cs = findmax(aprs) #[0]=Y,[1]=X
    xcb,ycb = xmins+cs[1],ymins+cs[0]
    print 'Initial guess for B location:',xcb,ycb

    ymins,ymaxs=ycc-4,ycc+4
    xmins,xmaxs=xcc-4,xcc+4
    aprs = image[ymins:ymaxs,xmins:xmaxs]
    #Find max pixel within aperture and call that pixel the initial guess:
    cs = findmax(aprs) #[0]=Y,[1]=X
    xcc,ycc = xmins+cs[1],ymins+cs[0]
    print 'Initial guess for C location:',xcc,ycc

    ymins,ymaxs=yca-4,yca+4
    xmins,xmaxs=xca-4,xca+4
    aprs = image[ymins:ymaxs,xmins:xmaxs]
    #Find max pixel within aperture and call that pixel the initial guess:
    cs = findmax(aprs) #[0]=Y,[1]=X
    xca,yca = xmins+cs[1],ymins+cs[0]
    print 'Initial guess for A location:',xca,yca

    # Write out initial guess to file:
    os.system('touch '+folder+line.split('.')[2]+ '_initialguess')
    newfile = directory+'/'+line.split('.')[2]+ '_initialguess'
    print newfile
    #newfile = line.split('.')[2]+ '_initialguess'
    string = str(xca)+' '+str(yca)+' '+str(xcb)+' '+str(ycb)+' '+str(xcc)+' '+str(ycc)+' '+str(bkgdx)+' '+str(bkgdy)
    k = open(newfile, 'w')
    k.write(string + "\n")
    k.close()

os.system('rm list')
