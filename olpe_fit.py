import numpy as np
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Gaussian2D, Moffat2D
from astropy.io import fits
from astropy.io.fits import getheader
import os
import time as tm
import warnings
import csv

warnings.filterwarnings("ignore") #ignore runtime warnings given by masked math later in calculations

############################################## Definitions ##########################################
def proposal(value,valuewidth,arraysize):  #Function for generating flat distribution random variations
    new = np.random.normal(value,valuewidth,arraysize)
    return new
def logproposal(value,valuewidth,arraysize): #Function for generating log flat distribution random variations
    logvalue = np.log10(value)
    lognew = np.random.normal(logvalue,valuewidth,arraysize)
    new = 10**lognew
    return new
def findmax(data):  #Finds the pixel with the max value within the aperture centered at the orginal guess
    from numpy import unravel_index
    m = np.argmax(data)
    c = unravel_index(m, data.shape)
    return c
def sampler(pars):  #The model testing function
    xcs4,ycs4,xcc4,ycc4,amps14,ampc14,ampratio4,bkgdfill4,sigmax4,sigmay4,sigmax24,sigmay24,theta4,theta24,chi4 = pars[0],pars[1],pars[2],pars[3],\
      pars[4],pars[5],pars[6],pars[7],pars[8],pars[9],pars[10],pars[11],pars[12],pars[13],pars[14]
    chinew = []
    for xcs3,xcc3,ycs3,ycc3,amps13,ampc13,ampratio3,bkgd3,sigmax3,sigmay3,sigmax23,sigmay23,theta3,theta23 in zip(xcs4,xcc4,ycs4,ycc4,amps14,\
                                    ampc14,ampratio4,bkgdfill4,sigmax4,sigmay4,sigmax24,sigmay24,theta4,theta24):
        amps23=amps13*ampratio3
        amps3=amps13-amps23
        psfs1 = models.Gaussian2D(amplitude = amps3, x_mean=xcs3, y_mean=ycs3, x_stddev=sigmax3, y_stddev=sigmay3, theta=theta3)
        psfs2 = models.Gaussian2D(amplitude = amps23, x_mean=xcs3, y_mean=ycs3, x_stddev=sigmax23, y_stddev=sigmay23, theta=theta23)
        psfs = psfs1(x,y)+psfs2(x,y)
        ampc23=ampc13*ampratio3
        ampc3=ampc13-ampc23
        psfc1 = models.Gaussian2D(amplitude = ampc3, x_mean=xcc3, y_mean=ycc3, x_stddev=sigmax3, y_stddev=sigmay3, theta=theta3)
        psfc2 = models.Gaussian2D(amplitude = ampc23, x_mean=xcc3, y_mean=ycc3, x_stddev=sigmax23, y_stddev=sigmay23, theta=theta23)
        psfc = psfc1(x,y)+psfc2(x,y)
        bkgd = np.ndarray(shape=image.shape, dtype=float)
        bkgd.fill(bkgd3)
        psf = psfs + psfc + bkgd
        chi1 = ((image-psf)/err)**2
        chi1 = np.sum(chi1)
        chinew.append(chi1)
    
    chinew=np.array(chinew)
    #print 'chinew ',chinew
    #print 'old chi ',chi4
    accept = np.where(chinew<chi4)
    #print 'accept ',accept
    acc=[]
    for chinew1,chi1 in zip(chinew,chi4):
        if chinew1<chi1:
            accept = 'yes'
            acc.append(accept)
        else:
            deltachi = chinew1-chi1
            #print deltachi
            p = np.exp(-deltachi/2.0)
            #print 'p',p
            dice = np.random.rand()
            #print 'dice',dice
            yesaccept = dice < p
            if yesaccept: 
                accept = 'yes'
                acc.append(accept)
                chi_return = chinew1
                probabilty_accepted='yes'
            else:
                accept='no'
                acc.append(accept)
                chi_return=chi1
                probabilty_accepted='no'
    #print 'acc ',acc
    return acc,chinew

########################################### Start code ######################################################
#############################################################################################################

#Define number of independant walkers you want
NWalkers = 12

# NIRC2 parameters:
FWHM=50 #mas
Pixscale = 9.95 #milliarcsec/pixel
FWHM = FWHM / Pixscale
sigma = FWHM/2.35

#### Image input:
print "Importing image..."
import argparse
# Get the file name from the entered argument
parser = argparse.ArgumentParser()
parser.add_argument("image_filename",type=str)
args = parser.parse_args()
filename=args.image_filename

image1 = fits.open(args.image_filename)
image = image1[0].data
imhdr = getheader(args.image_filename)

####  Look and see if there is already an mcmc started for this image.  If so, use the current value of the parameters
# as the initial value.  If not, begin the normal procedure for finding initial values.
directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results'
exists = os.path.exists(directory) #gives 'true' if the path exists, 'false' if it does not.
newdir=directory

if exists:
    print "An MCMC has already been started for this image."
else:
    #### Make new directory using the image file name to store results
    newdir = directory
    makedir = 'mkdir '+ newdir
    try:
        os.system(makedir)
    except:
        pass

# Check to see if each parameter has already written out arrays from previous trials.  If not, assign initial
# values for parameters
xcsexists = os.path.exists(directory+'/'+'xcsarray.csv')
ycsexists = os.path.exists(directory+'/'+'ycsarray.csv')
xccexists = os.path.exists(directory+'/'+'xccarray.csv')
yccexists = os.path.exists(directory+'/'+'yccarray.csv')
if xcsexists and ycsexists and xccexists and yccexists:
    print "Taking old position varibles"
    b = open(newdir+'/'+'xcsarray.csv', 'r')
    c = b.read()
    # read in the csv file, split the rows (the filter removes any empty lines):
    d = filter(None,c.split('\n'))
    # for each row ("i in range(len(d))"), split the row by commas, make each object a float, and place
    # them into a numpy array.  Now each walker is callable as arrays within this array
    xcsarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    #xcs_in = [np.mean(xcs_in[i]) for i in range(NWalkers)] #Find the mean in each walker
    #xcs = np.mean(xcs_in) #Find the mean of the means in each walker and take this as the initial value for xcs parameter.
    # Check that the previous run had the same NWalkers as this one:
    if len(xcsarray) != NWalkers:
        print 'The previous MCMC used a different number of walkers.  To use this previous data, adjust NWalkers to ',len(xcs)
        exit()
    b = open(newdir+'/'+'ycsarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    ycsarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    b = open(newdir+'/'+'xccarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    xccarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    b = open(newdir+'/'+'yccarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    yccarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]

    # Take the last value in each parameter array for each walker to use as the value to make the initial models.  Written so that it works with
    # unknown number of values in each walker, so that it can be used with previously run MCMCs or brand new ones:
    xcs = [xcsarray[i][len(xcsarray[0])-1] for i in range(NWalkers)]
    ycs = [ycsarray[i][len(ycsarray[0])-1] for i in range(NWalkers)]
    xcc = [xccarray[i][len(xccarray[0])-1] for i in range(NWalkers)]
    ycc = [yccarray[i][len(yccarray[0])-1] for i in range(NWalkers)]

else:  
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

    # Expand initial guess to array that is the shape of the number of walkers
    xcs,ycs,xcc,ycc = [xcs]*NWalkers,[ycs]*NWalkers,[xcc]*NWalkers,[ycc]*NWalkers
    # ^These are now python lists of dimensions NWalkers, so that they can be fed into the initial model guess

    # Initialize parameter arrays:
    xcsarray = [xcs[0] for i in range(NWalkers)]
    ycsarray = [ycs[0] for i in range(NWalkers)]
    xccarray = [xcc[0] for i in range(NWalkers)]
    yccarray = [ycc[0] for i in range(NWalkers)]

ampsexists = os.path.exists(directory+'/'+'ampsarray.csv')
if ampsexists:
    print 'Taking old amps'
    b = open(newdir+'/'+'ampsarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    ampsarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    amps = [ampsarray[i][len(ampsarray[0])-1] for i in range(NWalkers)]
else:
    # 1 = Narrow gaussian; 2 = wide gaussian
    #Amplitudes initial guess:
    amps1=image[yms,xms] #max pixel value for star
    amps2=amps1*0.2 #emperically determined decent fit for initial guess
    amps = amps1-amps2 #amplitude of narrow gaussian is the max pixel data minus to amp of wide gaussian, so added
    #together they match the max pixel value of the star
    amps = [amps]*NWalkers
    ampsarray = [amps[0] for i in range(NWalkers)] # initialize parameter array

ampcexists = os.path.exists(directory+'/'+'ampcarray.csv')
if ampcexists:
    print 'Taking old ampc'
    b = open(newdir+'/'+'ampcarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    ampcarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    ampc = [ampcarray[i][len(ampcarray[0])-1] for i in range(NWalkers)]
else:
    ampc1=image[ymc,xmc] #max pixel value of companion
    ampc2=ampc1*0.2
    ampc=ampc1-ampc2
    ampc = [ampc]*NWalkers
    ampcarray = [ampc[0] for i in range(NWalkers)]

ampratioexists = os.path.exists(directory+'/'+'ampratioarray.csv')
if ampratioexists:
    print 'Taking old ampratio'
    b = open(newdir+'/'+'ampratioarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    ampratioarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    ampratio = [ampratioarray[i][len(ampratioarray[0])-1] for i in range(NWalkers)]
else:
    ampratio = [0.2]*NWalkers
    ampratioarray = [ampratio[0] for i in range(NWalkers)]
    
bkgdexists = os.path.exists(directory+'/'+'bkgdarray.csv')
if bkgdexists:
    print 'Taking old bkgd'
    b = open(newdir+'/'+'bkgdarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    bkgdarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    bkgdfill = [bkgdarray[i][len(bkgdarray[0])-1] for i in range(NWalkers)]
else:
    bkgdfill = [1.0]*NWalkers
    bkgdarray=[bkgdfill[0] for i in range(NWalkers)]
    
sigmaxexists = os.path.exists(directory+'/'+'sigmaxarray.csv')
if sigmaxexists:
    print 'Taking old sigmax'
    b = open(newdir+'/'+'sigmaxarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    sigmaxarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    sigmax = [sigmaxarray[i][len(sigmaxarray[0])-1] for i in range(NWalkers)]
else:
    sigmax = [sigma]*NWalkers
    sigmaxarray=[sigmax[0] for i in range(NWalkers)]
    
sigmayexists = os.path.exists(directory+'/'+'sigmayarray.csv')
if sigmayexists:
    print 'Taking old sigmay'
    b = open(newdir+'/'+'sigmayarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    sigmayarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    sigmay = [sigmayarray[i][len(sigmayarray[0])-1] for i in range(NWalkers)]
else:
    sigmay = [sigma]*NWalkers
    sigmayarray=[sigmay[0] for i in range(NWalkers)]
    
sigmax2exists = os.path.exists(directory+'/'+'sigmax2array.csv')
if sigmax2exists:
    print 'Taking old sigmax2'
    b = open(newdir+'/'+'sigmax2array.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    sigmax2array = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    sigmax2 = [sigmax2array[i][len(sigmax2array[0])-1] for i in range(NWalkers)]
else:
    sigma2 = sigma*3 #emperically determined difference btwn narrow and wide
    sigmax2 = [sigma2]*NWalkers
    sigmax2array=[sigmax2[0] for i in range(NWalkers)]
    
sigmay2exists = os.path.exists(directory+'/'+'sigmay2array.csv')
if sigmay2exists:
    print 'Taking old sigmay2'
    b = open(newdir+'/'+'sigmay2array.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    sigmay2array = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    sigmay2 = [sigmay2array[i][len(sigmay2array[0])-1] for i in range(NWalkers)]
else:
    sigma2 = sigma*3
    sigmay2 = [sigma2]*NWalkers
    sigmay2array=[sigmay2[0] for i in range(NWalkers)]
    
thetaexists = os.path.exists(directory+'/'+'thetaarray.csv')
if thetaexists:
    print 'Taking old theta'
    b = open(newdir+'/'+'thetaarray.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    thetaarray = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    theta = [thetaarray[i][len(thetaarray[0])-1] for i in range(NWalkers)]
else:
    theta = [0.0]*NWalkers
    thetaarray=[theta[0] for i in range(NWalkers)]
    
theta2exists = os.path.exists(directory+'/'+'theta2array.csv')
if theta2exists:
    print 'Taking old theta2'
    b = open(newdir+'/'+'theta2array.csv', 'r')
    c = b.read()
    d = filter(None,c.split('\n'))
    theta2array = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
    theta2 = [theta2array[i][len(theta2array[0])-1] for i in range(NWalkers)]
else:
    theta2 = [0.0]*NWalkers
    theta2array=[theta[0] for i in range(NWalkers)]

###############################################################################################

# Initialize the chi array:
chi = []

####### Set bad pixels to 'nan' and mask them from the further calculations:
raw_dat = np.loadtxt(open("../nirc2.1024.1024.badpix","rb")) #Bad pixel list
raw_dat = raw_dat.astype(int)
#Create new zeros array:
shape = (1024,1024)
mask = np.zeros(shape) #make a mask array
for i in raw_dat:
    image[i[1]-1][i[0]-1] = 'nan' #set bad pixels to "nan"
for i in raw_dat:
    mask[i[1]-1][i[0]-1] = 1 #make a mask of the bad pixels
image = np.ma.masked_array(image, mask=mask)  #apply mask to image array

###################### Calculate the error in each pixel for the image:  #########################
## Readnoise error:
rnoise = np.ndarray(shape=(1024,1024), dtype=float) #create the array of the same shape as the image
coadds = float(imhdr['coadds'])
multisam = float(imhdr['multisam'])
sampmode = imhdr['sampmode']
if sampmode == 3.0:
    readnoise = (38.0/np.sqrt(multisam)) * (np.sqrt(coadds))
elif sampmode == 2.0:
    readnoise = 38 * (np.sqrt(coadds))
else: 
    readnoise = 38 * (np.sqrt(coadds))
rnoise.fill(readnoise)

## Poisson error:
gain = 4.0 #electrons per count
phot = image*gain
pois = np.sqrt(np.abs(phot))

#error: an array of the same size as the image containing the value of the error in each pixel
err = np.sqrt(rnoise**2+pois**2)

#### Make an entire synthetic image #####
#Make a grid the same size as the image to project the model image onto
y, x = np.mgrid[:1024,:1024]

print 'Making initial model guess...'

## Make initial models (all identical for now):
# The number of models should be NWalkers
chi = []

for xcs1,xcc1,ycs1,ycc1,amps11,ampc11,ampratio1,bkgd1,sigmax1,sigmay1,sigmax21,sigmay21,theta1,theta21 in zip(xcs,xcc,ycs,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2):
    amps21=amps11*ampratio1
    amps1=amps11-amps21
    psfs1 = models.Gaussian2D(amplitude = amps1, x_mean=xcs1, y_mean=ycs1, x_stddev=sigmax1, y_stddev=sigmay1, theta=theta1)
    psfs2 = models.Gaussian2D(amplitude = amps21, x_mean=xcs1, y_mean=ycs1, x_stddev=sigmax21, y_stddev=sigmay21, theta=theta21)
    psfs = psfs1(x,y)+psfs2(x,y)
    ampc21=ampc11*ampratio1
    ampc1=ampc11-ampc21
    psfc1 = models.Gaussian2D(amplitude = ampc1, x_mean=xcc1, y_mean=ycc1, x_stddev=sigmax1, y_stddev=sigmay1, theta=theta1)
    psfc2 = models.Gaussian2D(amplitude = ampc21, x_mean=xcc1, y_mean=ycc1, x_stddev=sigmax21, y_stddev=sigmay21, theta=theta21)
    psfc = psfc1(x,y)+psfc2(x,y)
    bkgd = np.ndarray(shape=image.shape, dtype=float)
    bkgd.fill(bkgd1)
    psf = psfs + psfc + bkgd
    chi1 = ((image-psf)/err)**2
    chi1 = np.sum(chi1)
    chi.append(chi1)

## Initialize parameters array:
#"parameters" is the master array of the 13 variables + chi^2 for each of the walkers.  It is therefore a 14xNWalkers array
parameters = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])

index = np.arange(0,NWalkers,1) #create indicies equal to the number of walkers

#initialize counts: (for tracking acceptance rate)
xcs_ac,ycs_ac,xcc_ac,ycc_ac,amps_ac,ampc_ac,ampratio_ac,bkgd_ac,sigmax_ac,sigmay_ac,sigmax2_ac,sigmay2_ac,theta_ac,theta2_ac = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
xcs_tot,ycs_tot,xcc_tot,ycc_tot,amps_tot,ampc_tot,ampratio_tot,bkgd_tot,sigmax_tot,sigmay_tot,sigmax2_tot,sigmay2_tot,theta_tot,theta2_tot = 0,0,0,0,0,0,0,0,0,0,0,0,0,0

# Index for use in iterating:
index = np.arange(0,NWalkers,1)

### Parameters width for guassian to draw guesses from:
positionswidth = 0.01 #pixels
positioncwidth = 0.3 #pixels
ampswidth = 0.0025 #log counts
ampcwidth = 0.05 #log counts
ampratiowidth = 0.001 #fraction of wide gaussian to narrow gaussian amplitude
bkgdwidth = 0.001 #log counts
sigmawidth = 0.002 #log pixels
sigma2width = 0.001 #log pixels
thetawidth = 0.008 #radians
theta2width = 0.01 #radians

Parameter_selector = ['xcs','ycs','xcc','ycc','amps','ampc','ampratio','bkgd','sigmax','sigmay','sigmax2','sigmay2','theta','theta2']

################################################### Begin loop ############################################################
print 'Beginning loop...'
start=tm.time()
count = 0
if xcsexists and ycsexists and xccexists and yccexists:
    accept_min=0 # Because if walkers were written out previously, the burn in was already accomplished in the previous run
else:
    accept_min=500 # This sets the burn in rate.
    
print 'accept_min:',accept_min
# Initialize the convergence variables and criteria:
xcsstd, ycsstd, xccstd, yccstd, ampsstd, ampcstd, ampratiostd, bkgdstd, sigmaxstd,sigmaystd,sigmax2std,sigmay2std,thetastd,theta2std  = 1.,1.,\
  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1. # initialize convergence variables
xcsconvergence,ycsconvergence,xccconvergence,yccconvergence,ampsconvergence,ampcconvergence,ampratioconvergence,bkgdconvergence,\
  sigmaxconvergence,sigmayconvergence,sigmax2convergence,sigmay2convergence,thetaconvergence,theta2convergence = 0.03,0.03,0.03,0.03,0.03,0.03,\
  0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03
xcscrossed,ycscrossed,xcccrossed,ycccrossed,ampscrossed,ampccrossed,ampratiocrossed,bkgdcrossed,sigmaxcrossed,sigmaycrossed,sigmax2crossed,\
  sigmay2crossed,thetacrossed,theta2crossed = 'False','False','False','False','False','False','False','False','False','False','False',\
  'False','False','False'
  
# Set convergence criteria:
while xcsstd >= xcsconvergence or ycsstd >= ycsconvergence or xccstd >= xccconvergence or yccstd >= yccconvergence or ampsstd >= ampsconvergence\
  or ampcstd >= ampcconvergence or ampratiostd >= ampratioconvergence or bkgdstd >= bkgdconvergence:
    count = count+1 #iterate the counter
    ## Gibbs sampler: randomly select which parameter to test on each loop:
    randn = np.random.randint(0,14)
    rand = Parameter_selector[randn]
    index = np.arange(0,NWalkers,1)
    #Reinitialize the variable names to the current values every loop:
    xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi = parameters[0],parameters[1],parameters[2],\
    parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],parameters[11],\
    parameters[12],parameters[13],parameters[14]
    #print parameters

    ## Now vary and test that parameter for goodness of fit:
    if rand == 'xcs':
        xcs_tot=xcs_tot+NWalkers #iterate the counter to track the number of times this parameter was tried
        #Sample a new value for xcs:
        xcsnew = proposal(xcs,positionswidth,NWalkers) 
        #Update the parameter array with the new value of xcs:
        params = np.array([xcsnew,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        #Perform the model generation and chi^2 determination with this new parameter array:
        acc=sampler(params)
        #Retrieve from the sampler whether or not this new model was a better fit than the old one, and the current
        #chi^2 value:
        accept,chinew = acc[0],acc[1]
        #If this was a better fit, replace the old parameters array with this new one with the new value:
        for accepted,xcsnew1,chinew1,i in zip(accept,xcsnew,chinew,index):
            if accepted == 'yes':
                xcs[i] = xcsnew1 #replace the value of xcs in the parameters array only if the new xcs is a better fit than the old
                chi[i] = chinew1 #replce the chi^2 value for that walker only if the new model was accepted
                xcs_ac = xcs_ac + 1
            elif accepted =='no':
                xcs[i] = xcs[i]
                chi[i] = chi[i]
        parameters[0]=xcs # update the master array with the new values of the chosen parameter and the new chi^2 for the models
        parameters[14]=chi
        # Place the accpeted new values into the parameter tracking array, keeping track of which walker accepted new values.  This for loop
        # places the new value in the array corresponding to the walker that generated it.  But it only does this once 10000 jumps on the
        # parameter have been accepted.  This gives some "burn in", so that positions far from the minimum don't influence the mean.
        if xcs_ac >= accept_min:
            for i in index:
                xcsarray[i]=np.append(xcsarray[i],parameters[0,i])
            xcsmeans = [np.mean(xcsarray[i]) for i in range(NWalkers)] # Array of the mean value in each walker
            xcsstd = np.std(xcsmeans) #Std dev between the means of each walker
        # write out walkers to a file (after burn in):
        if xcs_ac >= accept_min+50: # adding 50 gives it some time to build up a sequence to write out - csv writer fails if there is
            # only a single value in the array.
            try:  #In case there isn't a sequence yet for this variable, it won't quit the script.
                b = open(newdir+'/'+'xcsarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(xcsarray)
                b.close()
            except:
                pass

    elif rand == 'ycs':
        ycs_tot=ycs_tot+NWalkers 
        ycsnew = proposal(ycs,positionswidth,NWalkers) 
        params = np.array([xcs,ycsnew,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,ycsnew1,chinew1,i in zip(accept,ycsnew,chinew,index):
            if accepted == 'yes':
                ycs[i] = ycsnew1
                chi[i] = chinew1
                ycs_ac = ycs_ac + 1
            elif accepted =='no':
                ycs[i] = ycs[i]
                chi[i] = chi[i]
        parameters[1]=ycs
        parameters[14]=chi
        if ycs_ac >= accept_min:
            for i in index:
                ycsarray[i]=np.append(ycsarray[i],parameters[1,i])
            ycsmeans = [np.mean(ycsarray[i]) for i in range(NWalkers)] 
            ycsstd = np.std(ycsmeans) 
        if ycs_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'ycsarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(ycsarray)
                b.close()
            except:
                pass

    elif rand == 'xcc':
        xcc_tot=xcc_tot+NWalkers 
        xccnew = proposal(xcc,positioncwidth,NWalkers) 
        params = np.array([xcs,ycs,xccnew,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,xccnew1,chinew1,i in zip(accept,xccnew,chinew,index):
            if accepted == 'yes':
                xcc[i] = xccnew1
                chi[i] = chinew1
                xcc_ac = xcc_ac+1
            elif accepted =='no':
                xcc[i] = xcc[i]
                chi[i] = chi[i]
        parameters[2]=xcc
        parameters[14]=chi
        if xcc_ac >= accept_min: 
            for i in index:
                xccarray[i]=np.append(xccarray[i],parameters[2,i])
            xccmeans = [np.mean(xccarray[i]) for i in range(NWalkers)] 
            xccstd = np.std(xccmeans)
        if xcc_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'xccarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(xccarray)
                b.close()
            except:
                pass

    elif rand == 'ycc':
        ycc_tot=ycc_tot+NWalkers 
        yccnew = proposal(ycc,positioncwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,yccnew,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,yccnew1,chinew1,i in zip(accept,yccnew,chinew,index):
            if accepted == 'yes':
                ycc[i] = yccnew1
                chi[i] = chinew1
                ycc_ac = ycc_ac+1
            elif accepted =='no':
                ycc[i] = ycc[i]
                chi[i] = chi[i]
        parameters[3]=ycc
        parameters[14]=chi
        if ycc_ac >= accept_min:
            for i in index:
                yccarray[i]=np.append(yccarray[i],parameters[3,i])
            yccmeans = [np.mean(yccarray[i]) for i in range(NWalkers)] 
            yccstd = np.std(yccmeans)
        if ycc_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'yccarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(yccarray)
                b.close()
            except:
                pass

    elif rand == 'amps':
        amps_tot=amps_tot+NWalkers 
        ampsnew = logproposal(amps,ampswidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,ampsnew,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,ampsnew1,chinew1,i in zip(accept,ampsnew,chinew,index):
            if accepted == 'yes':
                amps[i] = ampsnew1
                chi[i] = chinew1
                amps_ac=amps_ac+1
            elif accepted =='no':
                amps[i] = amps[i]
                chi[i] = chi[i]
        parameters[4]=amps
        parameters[14]=chi
        if amps_ac >= accept_min:
            for i in index:
                ampsarray[i]=np.append(ampsarray[i],parameters[4,i])
            ampsmeans = [np.mean(ampsarray[i]) for i in range(NWalkers)] 
            ampsstd = np.std(ampsmeans)
        if amps_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'ampsarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(ampsarray)
                b.close()
            except:
                pass
            
    elif rand == 'ampc':
        ampc_tot=ampc_tot+NWalkers 
        ampcnew = logproposal(ampc,ampcwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampcnew,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,ampcnew1,chinew1,i in zip(accept,ampcnew,chinew,index):
            if accepted == 'yes':
                ampc[i] = ampcnew1
                chi[i] = chinew1
                ampc_ac=ampc_ac+1
            elif accepted =='no':
                ampc[i] = ampc[i]
                chi[i] = chi[i]
        parameters[5]=ampc
        parameters[14]=chi
        if ampc_ac >= accept_min:
            for i in index:
                ampcarray[i]=np.append(ampcarray[i],parameters[5,i])
            ampcmeans = [np.mean(ampcarray[i]) for i in range(NWalkers)] 
            ampcstd = np.std(ampcmeans)
        if ampc_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'ampcarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(ampcarray)
                b.close()
            except:
                pass

    elif rand == 'ampratio':
        ampratio_tot=ampratio_tot+NWalkers 
        amprationew = proposal(ampratio,ampratiowidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,amprationew,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,amprationew1,chinew1,i in zip(accept,amprationew,chinew,index):
            if accepted == 'yes':
                ampratio[i] = amprationew1
                chi[i] = chinew1
                ampratio_ac=ampratio_ac+1
            elif accepted =='no':
                ampratio[i] = ampratio[i]
                chi[i] = chi[i]
        parameters[6]=ampratio
        parameters[14]=chi
        if ampratio_ac >= accept_min:
            for i in index:
                ampratioarray[i]=np.append(ampratioarray[i],parameters[6,i])
            ampratiomeans = [np.mean(ampratioarray[i]) for i in range(NWalkers)] 
            ampratiostd = np.std(ampratiomeans)
        if ampratio_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'ampratioarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(ampratioarray)
                b.close()
            except:
                pass
            
    elif rand == 'bkgd':
        bkgd_tot=bkgd_tot+NWalkers 
        bkgdnew = logproposal(bkgdfill,bkgdwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdnew,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,bkgdnew1,chinew1,i in zip(accept,bkgdnew,chinew,index):
            if accepted == 'yes':
                bkgdfill[i] = bkgdnew1
                chi[i] = chinew1
                bkgd_ac = bkgd_ac+1
            elif accepted =='no':
                bkgdfill[i] = bkgdfill[i]
                chi[i] = chi[i]
        parameters[7]=bkgdfill
        parameters[14]=chi
        if bkgd_ac >= accept_min:
            for i in index:
                bkgdarray[i]=np.append(bkgdarray[i],parameters[7,i])
            bkgdmeans = [np.mean(bkgdarray[i]) for i in range(NWalkers)] 
            bkgdstd = np.std(bkgdmeans)
        if bkgd_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'bkgdarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(bkgdarray)
                b.close()
            except:
                pass
            
    elif rand == 'sigmax':
        sigmax_tot=sigmax_tot+NWalkers 
        sigmaxnew = logproposal(sigmax,sigmawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmaxnew,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,sigmaxnew1,chinew1,i in zip(accept,sigmaxnew,chinew,index):
            if accepted == 'yes':
                sigmax[i] = sigmaxnew1
                chi[i] = chinew1
                sigmax_ac=sigmax_ac+1
            elif accepted =='no':
                sigmax[i] = sigmax[i]
                chi[i] = chi[i]
        parameters[8]=sigmax
        parameters[14]=chi
        if sigmax_ac >= accept_min:
            for i in index:
                sigmaxarray[i]=np.append(sigmaxarray[i],parameters[8,i])
            sigmaxmeans = [np.mean(sigmaxarray[i]) for i in range(NWalkers)] 
            sigmaxstd = np.std(sigmaxmeans)
        if sigmax_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'sigmaxarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(sigmaxarray)
                b.close()
            except:
                pass
            
    elif rand == 'sigmay':
        sigmay_tot=sigmay_tot+NWalkers 
        sigmaynew = logproposal(sigmay,sigmawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmaynew,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,sigmaynew1,chinew1,i in zip(accept,sigmaynew,chinew,index):
            if accepted == 'yes':
                sigmay[i] = sigmaynew1
                chi[i] = chinew1
                sigmay_ac=sigmay_ac+1
            elif accepted =='no':
                sigmay[i] = sigmay[i]
                chi[i] = chi[i]
        parameters[9]=sigmay
        parameters[14]=chi
        if sigmay_ac >= accept_min:
            for i in index:
                sigmayarray[i]=np.append(sigmayarray[i],parameters[9,i])
            sigmaymeans = [np.mean(sigmayarray[i]) for i in range(NWalkers)] 
            sigmaystd = np.std(sigmaymeans)
        if sigmay_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'sigmayarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(sigmayarray)
                b.close()
            except:
                pass
            
    elif rand == 'sigmax2':
        sigmax2_tot=sigmax2_tot+NWalkers 
        sigmax2new = logproposal(sigmax2,sigma2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2new,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,sigmax2new1,chinew1,i in zip(accept,sigmax2new,chinew,index):
            if accepted == 'yes':
                sigmax2[i] = sigmax2new1
                chi[i] = chinew1
                sigmax2_ac=sigmax2_ac+1
            elif accepted =='no':
                sigmax2[i] = sigmax2[i]
                chi[i] = chi[i]
        parameters[10]=sigmax2
        parameters[14]=chi
        if sigmax2_ac >= accept_min:
            for i in index:
                sigmax2array[i]=np.append(sigmax2array[i],parameters[10,i])
            sigmax2means = [np.mean(sigmax2array[i]) for i in range(NWalkers)] 
            sigmax2std = np.std(sigmax2means)
        if sigmax2_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'sigmax2array.csv', 'w')
                a = csv.writer(b)
                a.writerows(sigmax2array)
                b.close()
            except:
                pass
            
    elif rand == 'sigmay2':
        sigmay2_tot=sigmay2_tot+NWalkers 
        sigmay2new = logproposal(sigmay2,sigma2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2new,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,sigmay2new1,chinew1,i in zip(accept,sigmay2new,chinew,index):
            if accepted == 'yes':
                sigmay2[i] = sigmay2new1
                chi[i] = chinew1
                sigmay2_ac=sigmay2_ac+1
            elif accepted =='no':
                sigmay2[i] = sigmay2[i]
                chi[i] = chi[i]
        parameters[11]=sigmay2
        parameters[14]=chi
        if sigmay2_ac >= accept_min:
            for i in index:
                sigmay2array[i]=np.append(sigmay2array[i],parameters[11,i])
            sigmay2means = [np.mean(sigmay2array[i]) for i in range(NWalkers)] 
            sigmay2std = np.std(sigmay2means)
        if sigmay2_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'sigmay2array.csv', 'w')
                a = csv.writer(b)
                a.writerows(sigmay2array)
                b.close()
            except:
                pass
            
    elif rand == 'theta':
        theta_tot=theta_tot+NWalkers 
        thetanew = proposal(theta,thetawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,thetanew,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,thetanew1,chinew1,i in zip(accept,thetanew,chinew,index):
            if accepted == 'yes':
                theta[i] = thetanew1
                chi[i] = chinew1
                theta_ac=theta_ac+1
            elif accepted =='no':
                theta[i] = theta[i]
                chi[i] = chi[i]
        parameters[12]=theta
        parameters[14]=chi
        if theta_ac >= accept_min:
            for i in index:
                thetaarray[i]=np.append(thetaarray[i],parameters[12,i])
            thetameans = [np.mean(thetaarray[i]) for i in range(NWalkers)] 
            thetastd = np.std(thetameans)
        if theta_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'thetaarray.csv', 'w')
                a = csv.writer(b)
                a.writerows(thetaarray)
                b.close()
            except:
                pass
            
    elif rand == 'theta2':
        theta2_tot=theta2_tot+NWalkers 
        theta2new = proposal(theta2,theta2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2new,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,theta2new1,chinew1,i in zip(accept,theta2new,chinew,index):
            if accepted == 'yes':
                theta2[i] = theta2new1
                chi[i] = chinew1
                theta2_ac=theta2_ac+1
            elif accepted =='no':
                theta2[i] = theta2[i]
                chi[i] = chi[i]
        parameters[13]=theta2
        parameters[14]=chi
        if theta2_ac >= accept_min:
            for i in index:
                theta2array[i]=np.append(theta2array[i],parameters[13,i])
            theta2means = [np.mean(theta2array[i]) for i in range(NWalkers)] 
            theta2std = np.std(theta2means)
        if theta2_ac >= accept_min+50:
            try:
                b = open(newdir+'/'+'theta2array.csv', 'w')
                a = csv.writer(b)
                a.writerows(theta2array)
                b.close()
            except:
                pass

    mod=count%10
    if mod==0:
        print 'Tested ',count,' loops, ', count*NWalkers, ' permutations...'

    mod2=count%100
    if mod2==0:# and xcs_ac >= accept_min:
        print ''
        print 'Acceptance rates:'
        print 'xcs:',xcs_ac,xcs_tot
        print 'ycs:',ycs_ac,ycs_tot
        print 'xcc:',xcc_ac,xcc_tot
        print 'ycc:',ycc_ac,ycc_tot
        print 'amps:',amps_ac,amps_tot
        print 'ampc:',ampc_ac,ampc_tot
        print 'ampratio:',ampratio_ac,ampratio_tot
        print 'bkgd:',bkgd_ac,bkgd_tot

    # Setting the floating convergence criteria:
    mod3=count%100
    # Evaluating the overall scatter among accepted points (after burn in) to determine the convergence criteria
    if mod3 == 0 and xcs_ac >= accept_min:
        xcsmeanstd = np.std(xcsarray) # Determines the scatter of points within each walker
        xcsconvergence = 0.1 * xcsmeanstd # Sets the convergence criterion for this variable at 1/10th of the scatter among data points
        print ' ############ xcs convergence criteria: ',xcsconvergence,' ################'
    if mod3 == 0 and ycs_ac >= accept_min:
        ycsmeanstd = np.std(ycsarray)
        ycsconvergence = 0.1 * ycsmeanstd
        print ' ############ ycs convergence criteria: ',ycsconvergence,' ################'
    if mod3 == 0 and xcc_ac >= accept_min:
        xccmeanstd = np.std(xccarray)
        xccconvergence = 0.1 * xccmeanstd
        print ' ############ xcc convergence criteria: ',xccconvergence,' ################'
    if mod3 == 0 and ycc_ac >= accept_min:
        yccmeanstd = np.std(yccarray) 
        yccconvergence = 0.1 * yccmeanstd
        print ' ############ ycc convergence criteria: ',yccconvergence,' ################'
    if mod3 == 0 and amps_ac >= accept_min:
        ampsmeanstd = np.std(ampsarray) 
        ampsconvergence = 0.1 * ampsmeanstd
        print ' ############ amps convergence criteria: ',ampsconvergence,' ################'
    if mod3 == 0 and ampc_ac >= accept_min:
        ampcmeanstd = np.std(ampcarray) 
        ampcconvergence = 0.1 * ampcmeanstd
        print ' ############ ampc convergence criteria: ',ampcconvergence,' ################'
    if mod3 == 0 and ampratio_ac >= accept_min:
        ampratiomeanstd = np.std(ampratioarray) 
        ampratioconvergence = 0.1 * ampratiomeanstd
        print ' ############ ampratio convergence criteria: ',ampratioconvergence,' ##############'
    if mod3 == 0 and bkgd_ac >= accept_min:
        bkgdmeanstd = np.std(bkgdarray) 
        bkgdconvergence = 0.1 * bkgdmeanstd
        print ' ############ bkgd convergence criteria: ',bkgdconvergence,' ################'
        
    # Tracking when each variable has converged:
    if xcsstd < xcsconvergence:
        xcscrossed = 'True'
    else:
        xcscrossed = 'False'
    if ycsstd < ycsconvergence:
        ycscrossed = 'True'
    else:
        ycscrossed = 'False'
    if xccstd < xccconvergence:
        xcccrossed = 'True'
    else:
        xcccrossed = 'False'
    if yccstd < yccconvergence:
        ycccrossed = 'True'
    else:
        ycccrossed = 'False'
    if ampsstd < ampsconvergence:
        ampscrossed = 'True'
    else:
        ampscrossed = 'False'
    if ampcstd < ampcconvergence:
        ampccrossed = 'True'
    else:
        ampccrossed = 'False'
    if ampratiostd < ampratioconvergence:
        ampratiocrossed = 'True'
    else:
        ampratiocrossed = 'False'
    if bkgdstd < bkgdconvergence:
        bkgdcrossed = 'True'
    else:
        bkgdcrossed = 'False'

    if mod2==0:# and xcs_ac >= accept_min:    
        print ''
        print 'current xcs std between walkers:',xcsstd
        print 'current ycs std between walkers:',ycsstd
        print 'current xcc std between walkers:',xccstd
        print 'current ycc std between walkers:',yccstd
        print 'current amps std between walkers:',ampsstd
        print 'current ampc std between walkers:',ampcstd
        print 'current ampratio std between walkers:',ampratiostd
        print 'current bkgd std between walkers:',bkgdstd
        print ''
        print 'xcs converged: ',xcscrossed
        print 'ycs converged: ',ycscrossed
        print 'xcc converged: ',xcccrossed
        print 'ycc converged: ',ycccrossed
        print 'amps converged: ',ampscrossed
        print 'ampc converged: ',ampccrossed
        print 'ampratio converged: ',ampratiocrossed
        print 'bkgd converged: ',bkgdcrossed
    

print '...Done'
stop = tm.time()
time = (stop-start)/3600
print 'This operation took ',time,' hours'
from datetime import date
z = open(newdir+'/'+'log', 'a')
string = str(date.today())
string += str(count) +'loops and '+str(count*NWalkers)+'permutations'+ "\n"
string += 'xcs walker mean:'+str(np.mean(xcsmeans))+ "\n"
string += 'xcs walker std:'+str(xcsstd)+ "\n"
string += 'ycs walker mean:'+str(np.mean(ycsmeans))+ "\n"
string += 'ycs walker std:'+str(ycsstd)+ "\n"
string += 'xcc walker mean:'+str(np.mean(xccmeans))+ "\n"
string += 'xcc walker std:'+str(xccstd)+ "\n"
string += 'ycc walker mean:'+str(np.mean(yccmeans))+ "\n"
string += 'ycc walker std:'+str(yccstd)+ "\n"
string += 'Acceptance rates:'+'\n'
string += 'xcs:'+str(xcs_ac)+','+str(xcs_tot)
string += 'ycs:'+str(ycs_ac)+','+str(ycs_tot)
string += 'xcc:'+str(xcc_ac)+','+str(xcc_tot)
string += 'ycc:'+str(ycc_ac)+','+str(ycc_tot)
string += 'amps:'+str(amps_ac)+','+str(amps_tot)
string += 'ampc:'+str(ampc_ac)+','+str(ampc_tot)
string += 'ampratio:'+str(ampratio_ac)+','+str(ampratio_tot)
string += 'bkgd:'+str(bkgd_ac)+','+str(bkgd_tot)
string += 'sigmax:'+str(sigmax_ac)+','+str(sigmax_tot)
string += 'sigmay:'+str(sigmay_ac)+','+str(sigmay_tot)
string += 'sigmax2:'+str(sigmax2_ac)+','+str(sigmax2_tot)
string += 'sigmay2:'+str(sigmay2_ac)+','+str(sigmay2_tot)
string += 'theta:'+str(theta_ac)+','+str(theta_tot)
string += 'theta2:'+str(theta2_ac)+','+str(theta2_tot)
string += 'This operation took '+str(time)+' hours'
z.write(string + "\n")
z.close()
