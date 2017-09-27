import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import argparse

######
## Make a list of image files in the folder:

parser = argparse.ArgumentParser()
parser.add_argument("folder",type=str)
args = parser.parse_args()
folder=args.folder
directory = folder.split('/')[0]+'/'+folder.split('/')[1]+'/'

os.system('ls '+directory+'*.fits > list')

def findmax(data):  #Finds the pixel with the max value within the aperture centered at the orginal guess
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
    print 'Press "D" key to select center of companion'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',vmin=np.percentile(image,5),vmax=np.percentile(image,95))
    ax.set_title('Hover mouse over companion and type D')
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
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()
    print coordc
    coordc=coordc[0]
    xmc,ymc = coordc[0],coordc[1]
    xmc,ymc = int(xmc),int(ymc)

    print 'Press "D" key to select center of star'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower')
    ax.set_title('Hover mouse over star and type D')
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
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()
    print coords
    coords=coords[0]
    xms,yms = coords[0],coords[1]
    xms,yms = int(xms),int(yms)

    print 'Press "D" key to select backgroung smaple'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap='gray',origin='lower',vmin=np.percentile(image,5),vmax=np.percentile(image,95))
    ax.set_title('Hover mouse over an empty sky area and type D')
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
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
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
    filename=line
    newfile = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_initial_position_guess'
    string = str(xcs)+' '+str(ycs)+' '+str(xcc)+' '+str(ycc)+' '+str(bkgdx)+' '+str(bkgdy)
    k = open(newfile, 'a')
    k.write(string + "\n")
    k.close()

