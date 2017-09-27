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
        amps13=amps13-bkgd3
        amps23=amps13*ampratio3
        amps3=amps13-amps23
        psfs1 = models.Gaussian2D(amplitude = amps3, x_mean=xcs3, y_mean=ycs3, x_stddev=sigmax3, y_stddev=sigmay3, theta=theta3)
        psfs2 = models.Gaussian2D(amplitude = amps23, x_mean=xcs3, y_mean=ycs3, x_stddev=sigmax23, y_stddev=sigmay23, theta=theta23)
        psfs = psfs1(x,y)+psfs2(x,y)
        ampc13=ampc13-bkgd3
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
NWalkers = 1

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
    print "Taking in output from Step 1 or previously started MCMC."
else:
    #### Make new directory using the image file name to store results
    newdir = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results'
    makedir = 'mkdir '+ newdir
    os.system(makedir)

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
    #### Import initial positions guess:
    fileguess = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_initial_position_guess'
    positionguess = np.loadtxt(open(fileguess,"rb"),delimiter=' ')
    xcs,ycs,xcc,ycc = positionguess[0],positionguess[1],positionguess[2],positionguess[3]
    
    # Integers of initial position guess for determining amplitude initial guess:
    xms,yms,xmc,ymc = xcs-0.5,ycs-0.5,xcc-0.5,ycc-0.5
    xms,yms,xmc,ymc = int(xms),int(yms),int(xmc),int(ymc)

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
    amps=image[yms,xms] #max pixel value for star
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
    ampc=image[ymc,xmc] #max pixel value of companion
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
    ampratio = [0.2]*NWalkers #Emperically determined to be a godd estimate of wide to narrow gaussian amplitude ratio
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
    fileguess = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_initial_position_guess'
    positionguess = np.loadtxt(open(fileguess,"rb"),delimiter=' ')
    back_x,back_y = positionguess[4],positionguess[5]
    ymin_c = back_x
    ymax_c = back_x+10
    xmin_c = back_y
    xmax_c = back_y+10
    box = image[ymin_c:ymax_c,xmin_c:xmax_c]
    bkgdfill = np.mean(box)
    bkgdfill = [bkgdfill]*NWalkers
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
raw_dat = np.loadtxt(open("nirc2.1024.1024.badpix","rb")) #Bad pixel list
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
    amps11 = amps11 - bkgd1 #Subtract off the sky background from model amplitude
    amps21=amps11*ampratio1 #Amplitude of wide gaussian is a fraction of the total amplitude
    amps1=amps11-amps21 #Amplitude of narrow gaussian is the total amplitude minus the narrow gaussian
    psfs1 = models.Gaussian2D(amplitude = amps1, x_mean=xcs1, y_mean=ycs1, x_stddev=sigmax1, y_stddev=sigmay1, theta=theta1)
    psfs2 = models.Gaussian2D(amplitude = amps21, x_mean=xcs1, y_mean=ycs1, x_stddev=sigmax21, y_stddev=sigmay21, theta=theta21)
    psfs = psfs1(x,y)+psfs2(x,y)
    ampc11 = ampc11 - bkgd1
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
accept_min=0

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
convergenceratio = 0.05 # This is the fraction of the scatter among the data points that will be the convergence criteria for scatter between the walker means.  0.1 means scatter between the means that is 1/10th of the scatter among the data points will determine convergence is met.

burn_in=4500 # This sets how many trials for each variable in step 2.

# Run the loop until all parameters have been sampled a minimum number of times:
while xcs_tot <= burn_in or ycs_tot <= burn_in or xcc_tot <= burn_in or ycc_tot <= burn_in or amps_tot <= burn_in\
  or ampc_tot <= burn_in or ampratio_tot <= burn_in or bkgd_tot <= burn_in or sigmax_tot <= burn_in or sigmay_tot <= burn_in\
  or sigmax2_tot <= burn_in or sigmay2_tot <= burn_in or theta_tot <= burn_in or theta2_tot <= burn_in:
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
            # write out value to a file:
            b = open(newdir+'/'+'xcsarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(xcsarray)
            b.close()

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
            b = open(newdir+'/'+'ycsarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(ycsarray)
            b.close()

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
            b = open(newdir+'/'+'xccarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(xccarray)
            b.close()
            
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
            b = open(newdir+'/'+'yccarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(yccarray)
            b.close()
       
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
            b = open(newdir+'/'+'ampsarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(ampsarray)
            b.close()
            
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
            b = open(newdir+'/'+'ampcarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(ampcarray)
            b.close()

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
            b = open(newdir+'/'+'ampratioarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(ampratioarray)
            b.close()
            
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
            b = open(newdir+'/'+'bkgdarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(bkgdarray)
            b.close()
            
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
            b = open(newdir+'/'+'sigmaxarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(sigmaxarray)
            b.close()
            
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
            b = open(newdir+'/'+'sigmayarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(sigmayarray)
            b.close()
        
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
            b = open(newdir+'/'+'sigmax2array_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(sigmax2array)
            b.close()
            
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
            b = open(newdir+'/'+'sigmay2array_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(sigmay2array)
            b.close()

            
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
            b = open(newdir+'/'+'thetaarray_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(thetaarray)
            b.close()
        
            
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
            b = open(newdir+'/'+'theta2array_step2.csv', 'w')
            a = csv.writer(b)
            a.writerows(theta2array)
            b.close()
       

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
        print ''
        print 'Current values:'
        print 'xcs:',parameters[0]
        print 'ycs:',parameters[1]
        print 'xcc:',parameters[2]
        print 'ycc:',parameters[3]
        print 'amps:',parameters[4]
        print 'ampc:',parameters[5]
        print 'ampratio:',parameters[6]
        print 'bkgd:',parameters[7]
        
        

    

print '...Done'
stop = tm.time()
time = (stop-start)/3600
print 'This operation took ',time,' hours'


### Take the last value of each parameters and output it to a file:
xcs_out = xcsarray[0][len(xcsarray[0])-1]
ycs_out = ycsarray[0][len(ycsarray[0])-1]
xcc_out = xccarray[0][len(xccarray[0])-1]
ycc_out = yccarray[0][len(yccarray[0])-1]
amps_out = ampsarray[0][len(ampsarray[0])-1]
ampc_out = ampcarray[0][len(ampcarray[0])-1]
ampratio_out = ampratioarray[0][len(ampratioarray[0])-1]
bkgd_out = bkgdarray[0][len(bkgdarray[0])-1]
sigmax_out = sigmaxarray[0][len(sigmaxarray[0])-1]
sigmay_out = sigmayarray[0][len(sigmayarray[0])-1]
sigmax2_out = sigmax2array[0][len(sigmax2array[0])-1]
sigmay2_out = sigmay2array[0][len(sigmay2array[0])-1]
theta_out = thetaarray[0][len(thetaarray[0])-1]
theta2_out = theta2array[0][len(theta2array[0])-1]


z = open(newdir+'/'+filename.split('.')[2]+'_step2_output', 'w')
string = str(xcs_out) + ' , ' + str(ycs_out) + ' , ' + str(xcc_out) + ' , ' + str(ycc_out) + ' , ' + str(amps_out) + ' , ' + str(ampc_out)\
   + ' , ' + str(ampratio_out) + ' , ' + str(bkgd_out) + ' , ' + str(sigmax_out) + ' , ' + str(sigmay_out) + ' , ' + str(sigmax2_out) + ' , ' \
   +  str(sigmay2_out) + ' , ' + str(theta_out) + ' , ' + str(theta2_out)
z.write(string + "\n")
z.close()
