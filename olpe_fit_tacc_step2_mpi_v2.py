# This is an MCMC fitting algorithm built for astrometric position finding of two objects in NIRC2 images.  It is designed 
# as part of a pipeline for finding the relative position angle and separation of a candidate companion around a host star using
# the Lonestar 5 TACC supercomputer.
# This is step 2 of the OLPE Fit pipeline.
# For more information consult the OLPE Fit readme file.
#
# Input:
#  - NIRC2 image file
#  - Output file from step 1 titled 'filename_initial_position_guess' located in the same directory as the image
#       - this gives an initial guess for x/y positions of star and companion, an background levels.
#  - NIRC2 bad pixel list titled "nirc2.1024.1024.badpix" located in same directoy as this file (optional)
#
# Output:
#  - A folder containing .csv files of the entire MCMC chain for each subprocess titled "(rank)_finalarray_mpi.csv"
#       titled "filename_olpefit_results_mpi"
#
# From the terminal (not on TACC), execute as follows:
# mpiexec -n xx python olpe_fit_tacc_step2_mpi.py path_to_image_file
# xx = number of cores
# (for TACC execution instructions consult readme file)
#
# Written by Logan A. Pearce

#####################################################################################################
######################################### Begin script ##############################################
#####################################################################################################

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
##MPI STUFF
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

warnings.filterwarnings("ignore") #ignore runtime warnings given by masked math later in calculations

#######################################################################################################
######################################### Definitions #################################################
#######################################################################################################

def proposal(value,valuewidth,arraysize):  #Function for generating normal distribution random variations
    new = np.random.normal(value,valuewidth,arraysize)
    return new
def logproposal(value,valuewidth,arraysize): #Function for generating log normal distribution random variations
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
    xcs4,ycs4,xcc4,ycc4,dx4,dy4,amps14,ampc14,ampratio4,bkgdfill4,sigmax4,sigmay4,sigmax24,sigmay24,theta4,theta24,chi4 = pars[0],pars[1],pars[2],\
      pars[3],pars[4],pars[5],pars[6],pars[7],pars[8],pars[9],pars[10],pars[11],pars[12],pars[13],pars[14],pars[15],pars[16]
    chinew = []
    for xcs3,xcc3,ycs3,ycc3,dx3,dy3,amps13,ampc13,ampratio3,bkgd3,sigmax3,sigmay3,sigmax23,sigmay23,theta3,theta23 in zip(xcs4,xcc4,ycs4,ycc4,\
                        dx4,dy4,amps14,ampc14,ampratio4,bkgdfill4,sigmax4,sigmay4,sigmax24,sigmay24,theta4,theta24):
        amps13=amps13-bkgd3
        amps23=amps13*ampratio3
        amps3=amps13-amps23
        xcs23,ycs23,xcc23,ycc23=xcs3+dx3,ycs3+dy3,xcc3+dx3,ycc3+dy3
        psfs1 = models.Gaussian2D(amplitude = amps3, x_mean=xcs3, y_mean=ycs3, x_stddev=sigmax3, y_stddev=sigmay3, theta=theta3)
        psfs2 = models.Gaussian2D(amplitude = amps23, x_mean=xcs23, y_mean=ycs23, x_stddev=sigmax23, y_stddev=sigmay23, theta=theta23)
        psfs = psfs1(x,y)+psfs2(x,y)
        ampc13=ampc13-bkgd3
        ampc23=ampc13*ampratio3
        ampc3=ampc13-ampc23
        psfc1 = models.Gaussian2D(amplitude = ampc3, x_mean=xcc3, y_mean=ycc3, x_stddev=sigmax3, y_stddev=sigmay3, theta=theta3)
        psfc2 = models.Gaussian2D(amplitude = ampc23, x_mean=xcc23, y_mean=ycc23, x_stddev=sigmax23, y_stddev=sigmay23, theta=theta23)
        psfc = psfc1(x,y)+psfc2(x,y)
        bkgd = np.ndarray(shape=image.shape, dtype=float)
        bkgd.fill(bkgd3)
        psf = psfs + psfc + bkgd
        chi1 = ((image-psf)/err)**2
        chi1 = np.sum(chi1)
        chinew.append(chi1)
    
    chinew=np.array(chinew)
    accept = np.where(chinew<chi4)
    acc=[]
    for chinew1,chi1 in zip(chinew,chi4):
        if chinew1<chi1:
            accept = 'yes'
            acc.append(accept)
        else:
            deltachi = chinew1-chi1
            p = np.exp(-deltachi/2.0)
            dice = np.random.rand()
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
    return acc,chinew


#######################################################################################################
#################################### User defined settings: ###########################################
#######################################################################################################
#                                                                                                     #
#       Change the value of these variables to set the MCMC to desired settings:                      #
#           -accept_min: after this many loop counts, the arrays will begin outputting values.        #
#           -NWalkers: the number of walkers desired for each subprocess to use.  The total           #
#                 number of walkers will be NWalkers*ncor                                             #
#           -totalcount: the total number of jumps the MCMC will execute                              #
#           -convergenceratio: the fraction of the scatter among the data points that will be         #
#                 the convergence criteria for scatter between the walker means if using more         #
#                 than one walker per subprocess                                                      #
#                 (note - comments must be adjusted at the while loop if using convergence ratio)     #
#           -useconvergence: tell OLPE Fit if you want to use convergence criteria or a simple count  #
#                 set to "yes" if using more than one walker per subprocess                           #
#                                                                                                     #
#######################################################################################################

accept_min= 0
NWalkers = 1
totalcount = 25000
convergenceratio = 0.05
useconvergence = 'no'

#######################################################################################################
###################### Input image and establish initial parameter values #############################
#######################################################################################################

# define the communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ncor = size

# NIRC2 parameters:
FWHM=50 #mas
Pixscale = 9.95 #milliarcsec/pixel - this is an approximate value for FWHM purposes only.
FWHM = FWHM / Pixscale
sigma = FWHM/2.35

############################ Input image: ############################
if rank ==0:
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

xsize,ysize = image.shape[1],image.shape[0]

#  Look and see if there is already an mcmc started for this image.  If so, use the current value of the parameters
# as the initial value.  If not, begin the normal procedure for finding initial values:
directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_mpi'
exists = os.path.exists(directory) #gives 'true' if the path exists, 'false' if it does not.
newdir=directory

if exists and rank==0:
    print "Taking in output from Step 1 or previously started MCMC."
else:
    #### Make new directory using the image file name to store results
    newdir = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_mpi'
    makedir = 'mkdir '+ newdir
    os.system(makedir)

# Check to see if each parameter has already written out arrays from previous trials.  If not, assign initial
# values for parameters
paramsexists = os.path.exists(directory+'/0_finalarray_mpi.csv')

############################# This part doesn't quite work yet ###############################################
if paramsexists:
    if rank == 0:
        print 'An MCMC has already been started for this image.  Taking previous parameter values.'
    # If there has already been an mcmc started for this image, pull up the previous outputs corresponding to
    # your processes' rank and use those arrays:
    c = np.genfromtxt('GSC6214/2016_06_16/40712_olpefit_results_mpi/'+str(rank)+'_finalarray_mpi.csv',delimiter=',')
    xcsarray,ycsarray,xccarray,yccarray =  [c[0] for i in range(NWalkers)],[c[1] for i in range(NWalkers)],[c[2] for i in range(NWalkers)],\
      [c[3] for i in range(NWalkers)]
    dxarray,dyarray =  [c[4] for i in range(NWalkers)],[c[5] for i in range(NWalkers)]
    ampsarray,ampcarray,ampratioarray,bkgdarray = [c[6] for i in range(NWalkers)],[c[7] for i in range(NWalkers)],\
      [c[8] for i in range(NWalkers)],[c[9] for i in range(NWalkers)]
    sigmaxarray,sigmayarray,sigmax2array,sigmay2array,thetaarray,theta2array,chiarray=[c[10] for i in range(NWalkers)],\
      [c[11] for i in range(NWalkers)],[c[12] for i in range(NWalkers)],[c[13] for i in range(NWalkers)],[c[14] for i in range(NWalkers)],\
      [c[15] for i in range(NWalkers)],[c[16] for i in range(NWalkers)]
    xcs = xcsarray[0][len(xcsarray[0])-1] # Take the final entry from the last run as the starting value for this run
    ycs,xcc,ycc = ycsarray[0][len(ycsarray)-1],xccarray[0][len(xccarray)-1],yccarray[0][len(yccarray)-1]
    dx,dy = dxarray[0][len(dxarray[0])-1],dyarray[0][len(dyarray[0])-1]
    amps,ampc,ampratio,bkgd = ampsarray[0][len(ampsarray)-1],ampcarray[0][len(ampcarray)-1],ampratioarray[0][len(ampratioarray)-1],\
      bkgdarray[0][len(bkgdarray)-1]
    sigmax,sigmay,sigmax2,sigmay2 = sigmaxarray[0][len(sigmaxarray)-1],sigmayarray[0][len(sigmayarray)-1],\
      sigmax2array[0][len(sigmax2array)-1],sigmay2array[0][len(sigmay2array)-1]
    theta,theta2 = thetaarray[0][len(thetaarray)-1],theta2array[0][len(theta2array)-1]
    if rank ==0:
        print xcs
        print type(xcs)
    # Expand to the shape of number of walkers
    xcs,ycs,xcc,ycc = [xcs]*NWalkers,[ycs]*NWalkers,[xcc]*NWalkers,[ycc]*NWalkers
    if rank ==0:
        print xcs[0]
    amps,ampc,ampratio,bkgd = [amps]*NWalkers,[ampc]*NWalkers,[ampratio]*NWalkers,[bkgd]*NWalkers
    sigmax,sigmay,sigmax2,sigmay2 = [sigmax]*NWalkers,[sigmay]*NWalkers,[sigmax2]*NWalkers,[sigmay2]*NWalkers
    theta,theta2 = [theta]*NWalkers,[theta2]*NWalkers
    bkgdfill = bkgd
    
    #Initialize parameters array
    parameters = c

    if rank ==0:
        print len(parameters[14])
    if rank ==0:
        print xcsarray
        print xcs
        print type(xcs)
        print len(xcs)

############ This part works fine #############
else:
    ## Make initial model guesses:
    if rank == 0:
        print 'Building initial model parameters...'
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

    # 1 = Narrow gaussian; 2 = wide gaussian
    # x- and y-offset for wide gaussian from narrow core:
    dx=[0]*NWalkers
    dy=[0]*NWalkers
    dxarray = [dx[0] for i in range(NWalkers)]
    dyarray = [dy[0] for i in range(NWalkers)]
    #Amplitudes initial guess:
    amps=image[yms,xms] #max pixel value for star
    amps = [amps]*NWalkers
    ampsarray = [amps[0] for i in range(NWalkers)]
    ampc=image[ymc,xmc] #max pixel value of companion
    ampc = [ampc]*NWalkers
    ampcarray = [ampc[0] for i in range(NWalkers)]
    ampratio = [0.2]*NWalkers #Emperically determined to be a godd estimate of wide to narrow gaussian amplitude ratio
    ampratioarray = [ampratio[0] for i in range(NWalkers)]
    
    # Background level
    fileguess = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_initial_position_guess'
    positionguess = np.loadtxt(open(fileguess,"rb"),delimiter=' ')
    back_x,back_y = int(positionguess[4]),int(positionguess[5])
    ymin_c = back_x
    ymax_c = back_x+10
    xmin_c = back_y
    xmax_c = back_y+10
    box = image[ymin_c:ymax_c,xmin_c:xmax_c]
    bkgdfill = np.mean(box)
    bkgdfill = [bkgdfill]*NWalkers
    bkgdarray=[bkgdfill[0] for i in range(NWalkers)]

    sigmax = [sigma]*NWalkers
    sigmaxarray=[sigmax[0] for i in range(NWalkers)]
    sigmay = [sigma]*NWalkers
    sigmayarray=[sigmay[0] for i in range(NWalkers)]
    sigma2 = sigma*3 #emperically determined difference btwn narrow and wide
    sigmax2 = [sigma2]*NWalkers
    sigmax2array=[sigmax2[0] for i in range(NWalkers)]
    sigma2 = sigma*3
    sigmay2 = [sigma2]*NWalkers
    sigmay2array=[sigmay2[0] for i in range(NWalkers)]

    theta = [0.0]*NWalkers
    thetaarray=[theta[0] for i in range(NWalkers)]
    theta2 = [0.0]*NWalkers
    theta2array=[theta[0] for i in range(NWalkers)]

    ## Initialize parameters array:
    #"parameters" is the master array of the 13 variables + chi^2 for each of the walkers.  It is therefore a 14xNWalkers array
    # parameters = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
    #paramsarray = parameters # initialize parameter tracking array
    #parametersfinal = paramsarray # initialize final  output array
    

#######################################################################################################
#################################### Mask bad pixels (optional) #######################################
#######################################################################################################

pixexists = os.path.exists("nirc2.1024.1024.badpix")
if pixexists:
    ####### Set bad pixels to 'nan' and mask them from the further calculations:
    if xsize == 1024 and ysize == 1024:
        raw_dat = np.loadtxt(open("nirc2.1024.1024.badpix","rb")) #Bad pixel list
    elif xsize == 512 and ysize == 512:
        raw_dat = np.loadtxt(open("nirc2.512.512.badpix","rb"))
        pass
    raw_dat = raw_dat.astype(int)
    #Create new zeros array:
    mask = np.zeros(image.shape) #make a mask array
    for i in raw_dat:
        image[i[1]-1][i[0]-1] = 'nan' #set bad pixels to "nan"
    for i in raw_dat:
        mask[i[1]-1][i[0]-1] = 1 #make a mask of the bad pixels
    image = np.ma.masked_array(image, mask=mask)  #apply mask to image array
else:
    pass

#######################################################################################################
####################### Calculate the error in each pixel for the image: ##############################
#######################################################################################################

# Readnoise error:
rnoise = np.ndarray(shape=image.shape, dtype=float) #create the array of the same shape as the image
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

# Poisson error:
gain = 4.0 #electrons per count
phot = image*gain
pois = np.sqrt(np.abs(phot))

#error: an array of the same size as the image containing the value of the error in each pixel
err = np.sqrt(rnoise**2+pois**2)

#######################################################################################################
############################### Make an initial model image: ##########################################
#######################################################################################################

#Make a grid the same size as the image to project the model image onto:
y, x = np.mgrid[:ysize,:xsize]

if rank ==0:
    print 'Making initial model guess...'

# Make initial models (all identical for now):
# The number of models should be NWalkers
chi = []

for xcs1,xcc1,ycs1,ycc1,dx1,dy1,amps11,ampc11,ampratio1,bkgd1,sigmax1,sigmay1,sigmax21,sigmay21,theta1,theta21 in zip(xcs,xcc,ycs,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2):
    amps11 = amps11 - bkgd1 #Subtract off the sky background from model amplitude
    amps21=amps11*ampratio1 #Amplitude of wide gaussian is a fraction of the total amplitude
    amps1=amps11-amps21 #Amplitude of narrow gaussian is the total amplitude minus the narrow gaussian
    xcs21,ycs21,xcc21,ycc21 = xcs1+dx1,ycs1+dy1,xcc1+dx1,ycc1+dy1 #x- and y-offset for center of wide gaussian from narrow - allows for
    #asymmetric Airy rings to be fit
    psfs1 = models.Gaussian2D(amplitude = amps1, x_mean=xcs1, y_mean=ycs1, x_stddev=sigmax1, y_stddev=sigmay1, theta=theta1)
    psfs2 = models.Gaussian2D(amplitude = amps21, x_mean=xcs21, y_mean=ycs21, x_stddev=sigmax21, y_stddev=sigmay21, theta=theta21)
    psfs = psfs1(x,y)+psfs2(x,y)
    ampc11 = ampc11 - bkgd1
    ampc21=ampc11*ampratio1
    ampc1=ampc11-ampc21
    psfc1 = models.Gaussian2D(amplitude = ampc1, x_mean=xcc1, y_mean=ycc1, x_stddev=sigmax1, y_stddev=sigmay1, theta=theta1)
    psfc2 = models.Gaussian2D(amplitude = ampc21, x_mean=xcc21, y_mean=ycc21, x_stddev=sigmax21, y_stddev=sigmay21, theta=theta21)
    psfc = psfc1(x,y)+psfc2(x,y)
    bkgd = np.ndarray(shape=image.shape, dtype=float)
    bkgd.fill(bkgd1)
    psf = psfs + psfc + bkgd
    chi1 = ((image-psf)/err)**2
    chi1 = np.sum(chi1)
    chi.append(chi1)

#######################################################################################################
######################### Initialize all tracking variables and arrays: ###############################
#######################################################################################################

# Initialize parameters array:
#"parameters" is the master array of the 15 variables + chi^2 for each of the walkers.  It is therefore a 14xNWalkers array
parameters = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
paramsarray = parameters # initialize parameter tracking array
parametersfinal = paramsarray # initialize final  output array

index = np.arange(0,NWalkers,1) #create indicies equal to the number of walkers

#initialize counts: (for tracking acceptance rate)
xcs_ac,ycs_ac,xcc_ac,ycc_ac,dx_ac,dy_ac,amps_ac,ampc_ac,ampratio_ac,bkgd_ac,sigmax_ac,sigmay_ac,sigmax2_ac,sigmay2_ac,theta_ac,theta2_ac = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
xcs_tot,ycs_tot,xcc_tot,ycc_tot,dx_tot,dy_tot,amps_tot,ampc_tot,ampratio_tot,bkgd_tot,sigmax_tot,sigmay_tot,sigmax2_tot,sigmay2_tot,theta_tot,theta2_tot = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

# Initialize the convergence variables and criteria:
xcsstd, ycsstd, xccstd, yccstd, dxstd, dystd, ampsstd, ampcstd, ampratiostd, bkgdstd, sigmaxstd,sigmaystd,sigmax2std,sigmay2std,thetastd,\
  theta2std  = 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1. # initialize convergence variables
xcsconvergence,ycsconvergence,xccconvergence,yccconvergence,dxconvergence,dyconvergence,ampsconvergence,ampcconvergence,ampratioconvergence,bkgdconvergence,sigmaxconvergence,sigmayconvergence,sigmax2convergence,sigmay2convergence,thetaconvergence,theta2convergence = 0.03,0.03,0.03,0.03,0.03,\
  0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03
xcscrossed,ycscrossed,xcccrossed,ycccrossed,dxcrossed,dycrossed,ampscrossed,ampccrossed,ampratiocrossed,bkgdcrossed,sigmaxcrossed,sigmaycrossed,\
  sigmax2crossed,sigmay2crossed,thetacrossed,theta2crossed = 'False','False','False','False','False','False','False','False','False','False',\
  'False','False','False','False','False','False'

# Index for use in iterating:
index = np.arange(0,NWalkers,1)

# Selector for Gibbs sampler:
Parameter_selector = ['xcs','ycs','xcc','ycc','dx','dy','amps','ampc','ampratio','bkgd','sigmax','sigmay','sigmax2','sigmay2','theta','theta2']

# Initialize the loop counter
count = 0 

#######################################################################################################
###################################### Set jump widths: ###############################################
#######################################################################################################

# (empirically determined)
positionswidth = 0.01 #pixels
positioncwidth = 0.3 #pixels
offsetxwidth = 0.08 #pixels
offsetywidth = 0.09 #pixels
ampswidth = 0.0025 #log counts
ampcwidth = 0.05 #log counts
ampratiowidth = 0.001 #fraction of wide gaussian to narrow gaussian amplitude
bkgdwidth = 0.001 #log counts
sigmawidth = 0.002 #log pixels
sigma2width = 0.001 #log pixels
thetawidth = 0.008 #radians
theta2width = 0.01 #radians

#######################################################################################################
###################################### Begin MCMC loop: ###############################################
#######################################################################################################

if rank ==0:
    print 'Beginning loop...'
start=tm.time()

# Run the loop until all parameters have been sampled a minimum number of times:
# If using walker convergence criteria to terminate, adjust comments accordingly

while count<=totalcount: #xcsstd >= xcsconvergence or ycsstd >= ycsconvergence or xccstd >= xccconvergence or yccstd >= yccconvergence or ampsstd >= ampsconvergence or ampcstd >= ampcconvergence or ampratiostd >= ampratioconvergence or bkgdstd >= bkgdconvergence:

    count = count+1 #iterate the counter
    ## Gibbs sampler: randomly select which parameter to test on each loop:
    randn = np.random.randint(0,16)
      #Because Python's intervals exclude the upper bound, this does not select the "chi" chain as a parameter to \
      #vary, which is the last row in the parameters array.
    rand = Parameter_selector[randn]
    
    #Reinitialize the variable names to the current values every loop:
    xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi = parameters[0],parameters[1],parameters[2],\
    parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10],parameters[11],\
    parameters[12],parameters[13],parameters[14],parameters[15],parameters[16]
    comm.barrier()
    
    ################# Now vary and test selected parameter for goodness of fit:  #################################
    if rand == 'xcs':
        xcs_tot=xcs_tot+NWalkers #iterate the counter to track the number of times this parameter was tried
        #Sample a new value for xcs:
        xcsnew = proposal(xcs,positionswidth,NWalkers) 
        #Update the parameter array with the new value of xcs:
        params = np.array([xcsnew,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[16]=chi
        # Place the accpeted new values into the parameter tracking array, keeping track of which walker accepted new values.  This for loop
        # places the new value in the array corresponding to the walker that generated it.  But it only does this once 10000 jumps on the
        # parameter have been accepted.  This gives some "burn in", so that positions far from the minimum don't influence the mean.
        if xcs_ac >= accept_min:
            for i in index:
                xcsarray[i]=np.append(xcsarray[i],parameters[0,i])
            xcsmeans = [np.mean(xcsarray[i]) for i in range(NWalkers)] # Array of the mean value in each walker
            xcsstd = np.std(xcsmeans) #Std dev between the means of each walker

    elif rand == 'ycs':
        ycs_tot=ycs_tot+NWalkers 
        ycsnew = proposal(ycs,positionswidth,NWalkers) 
        params = np.array([xcs,ycsnew,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[16]=chi
        if ycs_ac >= accept_min:
            for i in index:
                ycsarray[i]=np.append(ycsarray[i],parameters[1,i])
            ycsmeans = [np.mean(ycsarray[i]) for i in range(NWalkers)] 
            ycsstd = np.std(ycsmeans) 
           
    elif rand == 'xcc':
        xcc_tot=xcc_tot+NWalkers 
        xccnew = proposal(xcc,positioncwidth,NWalkers) 
        params = np.array([xcs,ycs,xccnew,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[16]=chi
        if xcc_ac >= accept_min: 
            for i in index:
                xccarray[i]=np.append(xccarray[i],parameters[2,i])
            xccmeans = [np.mean(xccarray[i]) for i in range(NWalkers)] 
            xccstd = np.std(xccmeans)
            
    elif rand == 'ycc':
        ycc_tot=ycc_tot+NWalkers 
        yccnew = proposal(ycc,positioncwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,yccnew,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[16]=chi
        if ycc_ac >= accept_min:
            for i in index:
                yccarray[i]=np.append(yccarray[i],parameters[3,i])
            yccmeans = [np.mean(yccarray[i]) for i in range(NWalkers)] 
            yccstd = np.std(yccmeans)

    elif rand == 'dx':
        dx_tot=dx_tot+NWalkers 
        dxnew = proposal(dx,offsetxwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dxnew,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,dxnew1,chinew1,i in zip(accept,dxnew,chinew,index):
            if accepted == 'yes':
                dx[i] = dxnew1
                chi[i] = chinew1
                dx_ac = dx_ac+1
            elif accepted =='no':
                dx[i] = dx[i]
                chi[i] = chi[i]
        parameters[4]=dx
        parameters[16]=chi
        if dx_ac >= accept_min:
            for i in index:
                dxarray[i]=np.append(dxarray[i],parameters[4,i])
            dxmeans = [np.mean(dxarray[i]) for i in range(NWalkers)] 
            dxstd = np.std(dxmeans)

    elif rand == 'dy':
        dy_tot=dy_tot+NWalkers 
        dynew = proposal(dy,offsetywidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dynew,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
        acc=sampler(params)
        accept,chinew = acc[0],acc[1]
        for accepted,dynew1,chinew1,i in zip(accept,dynew,chinew,index):
            if accepted == 'yes':
                dy[i] = dynew1
                chi[i] = chinew1
                dy_ac = dy_ac+1
            elif accepted =='no':
                dy[i] = dy[i]
                chi[i] = chi[i]
        parameters[5]=dy
        parameters[16]=chi
        if dy_ac >= accept_min:
            for i in index:
                dyarray[i]=np.append(dyarray[i],parameters[5,i])
            dymeans = [np.mean(dyarray[i]) for i in range(NWalkers)] 
            dystd = np.std(dymeans)
       
    elif rand == 'amps':
        amps_tot=amps_tot+NWalkers 
        ampsnew = logproposal(amps,ampswidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,ampsnew,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[6]=amps
        parameters[16]=chi
        if amps_ac >= accept_min:
            for i in index:
                ampsarray[i]=np.append(ampsarray[i],parameters[6,i])
            ampsmeans = [np.mean(ampsarray[i]) for i in range(NWalkers)] 
            ampsstd = np.std(ampsmeans)
            
    elif rand == 'ampc':
        ampc_tot=ampc_tot+NWalkers 
        ampcnew = logproposal(ampc,ampcwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampcnew,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[7]=ampc
        parameters[16]=chi
        if ampc_ac >= accept_min:
            for i in index:
                ampcarray[i]=np.append(ampcarray[i],parameters[7,i])
            ampcmeans = [np.mean(ampcarray[i]) for i in range(NWalkers)] 
            ampcstd = np.std(ampcmeans)
            
    elif rand == 'ampratio':
        ampratio_tot=ampratio_tot+NWalkers 
        amprationew = proposal(ampratio,ampratiowidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,amprationew,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[8]=ampratio
        parameters[16]=chi
        if ampratio_ac >= accept_min:
            for i in index:
                ampratioarray[i]=np.append(ampratioarray[i],parameters[8,i])
            ampratiomeans = [np.mean(ampratioarray[i]) for i in range(NWalkers)] 
            ampratiostd = np.std(ampratiomeans)
            
    elif rand == 'bkgd':
        bkgd_tot=bkgd_tot+NWalkers 
        bkgdnew = logproposal(bkgdfill,bkgdwidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdnew,sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[9]=bkgdfill
        parameters[16]=chi
        if bkgd_ac >= accept_min:
            for i in index:
                bkgdarray[i]=np.append(bkgdarray[i],parameters[9,i])
            bkgdmeans = [np.mean(bkgdarray[i]) for i in range(NWalkers)] 
            bkgdstd = np.std(bkgdmeans)
           
    elif rand == 'sigmax':
        sigmax_tot=sigmax_tot+NWalkers 
        sigmaxnew = logproposal(sigmax,sigmawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmaxnew,sigmay,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[10]=sigmax
        parameters[16]=chi
        if sigmax_ac >= accept_min:
            for i in index:
                sigmaxarray[i]=np.append(sigmaxarray[i],parameters[10,i])
            sigmaxmeans = [np.mean(sigmaxarray[i]) for i in range(NWalkers)] 
            sigmaxstd = np.std(sigmaxmeans)
            
    elif rand == 'sigmay':
        sigmay_tot=sigmay_tot+NWalkers 
        sigmaynew = logproposal(sigmay,sigmawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmaynew,sigmax2,sigmay2,theta,theta2,chi])
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
        parameters[11]=sigmay
        parameters[16]=chi
        if sigmay_ac >= accept_min:
            for i in index:
                sigmayarray[i]=np.append(sigmayarray[i],parameters[11,i])
            sigmaymeans = [np.mean(sigmayarray[i]) for i in range(NWalkers)] 
            sigmaystd = np.std(sigmaymeans)
           
    elif rand == 'sigmax2':
        sigmax2_tot=sigmax2_tot+NWalkers 
        sigmax2new = logproposal(sigmax2,sigma2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2new,sigmay2,theta,theta2,chi])
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
        parameters[12]=sigmax2
        parameters[16]=chi
        if sigmax2_ac >= accept_min:
            for i in index:
                sigmax2array[i]=np.append(sigmax2array[i],parameters[12,i])
            sigmax2means = [np.mean(sigmax2array[i]) for i in range(NWalkers)] 
            sigmax2std = np.std(sigmax2means)
            
    elif rand == 'sigmay2':
        sigmay2_tot=sigmay2_tot+NWalkers 
        sigmay2new = logproposal(sigmay2,sigma2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2new,theta,theta2,chi])
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
        parameters[13]=sigmay2
        parameters[16]=chi
        if sigmay2_ac >= accept_min:
            for i in index:
                sigmay2array[i]=np.append(sigmay2array[i],parameters[16,i])
            sigmay2means = [np.mean(sigmay2array[i]) for i in range(NWalkers)] 
            sigmay2std = np.std(sigmay2means)
            
    elif rand == 'theta':
        theta_tot=theta_tot+NWalkers 
        thetanew = proposal(theta,thetawidth,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,thetanew,theta2,chi])
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
        parameters[14]=theta
        parameters[16]=chi
        if theta_ac >= accept_min:
            for i in index:
                thetaarray[i]=np.append(thetaarray[i],parameters[14,i])
            thetameans = [np.mean(thetaarray[i]) for i in range(NWalkers)] 
            thetastd = np.std(thetameans)
            
    elif rand == 'theta2':
        theta2_tot=theta2_tot+NWalkers 
        theta2new = proposal(theta2,theta2width,NWalkers) 
        params = np.array([xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2new,chi])
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
        parameters[15]=theta2
        parameters[16]=chi
        if theta2_ac >= accept_min:
            for i in index:
                theta2array[i]=np.append(theta2array[i],parameters[15,i])
            theta2means = [np.mean(theta2array[i]) for i in range(NWalkers)] 
            theta2std = np.std(theta2means)

    ################## Collect the current state of each parameter into a chain to output to a file:  ############################
    paramsarray = np.hstack([paramsarray,parameters])
    
    comm.barrier() #wait for all processes to check in

    # every tenth loop have the root process print current status
    mod=count%10
    if mod==0 and rank==0:
        print 'Tested ',count,' loops, ', count*NWalkers, ' permutations...'
    comm.barrier()
    mod2=count%100
    if mod2==0 and rank==0 and xcs_ac >= accept_min:
        print ''
        print 'Acceptance rates:'
        print 'xcs:',xcs_ac,xcs_tot
        print 'ycs:',ycs_ac,ycs_tot
        print 'xcc:',xcc_ac,xcc_tot
        print 'ycc:',ycc_ac,ycc_tot
        print 'dx:',dx_ac,dx_tot
        print 'dy:',dy_ac,dy_tot
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
        print 'dx:',parameters[4]
        print 'dy:',parameters[5]
        print 'amps:',parameters[6]
        print 'ampc:',parameters[7]
        print 'ampratio:',parameters[8]
        print 'bkgd:',parameters[9]

    # Every 10th loop write the entire parameters array to a csv file.  It overwrites the previous file each time.
    mod2=count%10
    comm.barrier()
    if mod2==0 and count >= accept_min:
        finalarray = paramsarray
        b = open(newdir+'/'+str(rank)+'_finalarray_mpi.csv', 'w')
        a = csv.writer(b)
        a.writerows(finalarray)
        b.close()

    
        

    

print '...Done'
stop = tm.time()
time = (stop-start)/3600
print 'This operation took ',time,' hours'
