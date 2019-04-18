'''
############################ LAPF: Logan's Analytical PSF Fitter ##############################
                                          Step 2
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
#   Step 3: 
#
# usage (local): mpiexec -n number_of_processes python apf_step2.py ../1RXSJ1609/2009/N2.20090531.29966.LDIF.fits
        (tacc): sbatch script with the execute command: ibrun python apf_step2.py ../1RXSJ1609/2009/N2.20090531.29966.LDIF.fits
                  (for Lonestar 5, use 48 processes, for others use 24)

# User defined settings:
#    accept_min: script terminates when each parameter has been tried at least this many times.
#    burn_in: the first X amount of steps in the chain will be thrown away to constitute burn in

'''

accept_min = 100000
burn_in = 0

import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import Gaussian2D, Moffat2D
from astropy.io import fits
from astropy.io.fits import getheader
import os
import warnings
import csv
import argparse

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

# define the communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ncor = size

warnings.filterwarnings("ignore") #ignore runtime warnings given by masked math later in calculations

############################## define functions ################################################

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

##############
def build_2d_gaussian(xc,yc,dx,dy,total_amplitude,amplituderatio,background,narrow_sigma_x,narrow_sigma_y,\
                          wide_sigma_x,wide_sigma_y,narrow_theta,wide_theta):
    ''' Builds a 2d gaussian psf model from the sum of a wide and narrow gaussian.
        Inputs:
          xc/yc = x and y mean position
          dx/dy = position offset of center of wide gaussian from narrow gaussian
          total_amplitude = total amplitude of the psf, the brightest point of the object being modeled, 
            the sum of wide and narrow gaussians and background
          amplituderatio = ratio between amplitudes of wide to total amplitude
          background = background level
          sigmas = std dev of gaussian in x/y direction
          thetas = rotation of major axis
        Returns:
          a 2d Gaussian model on a grid
    '''
    # Make a grid the same size as the image:
    y, x = np.mgrid[:ysize,:xsize]
    total_amplitude = total_amplitude - background #subtract the background from the star's amplitude
    wide_amplitude = total_amplitude*amplituderatio #Amplitude of wide gaussian is a fraction of the total amplitude
    narrow_amplitude = total_amplitude - wide_amplitude #Amplitude of narrow gaussian is the total amplitude minus the narrow gaussian
    narrow_psf = models.Gaussian2D(amplitude = narrow_amplitude, x_mean = xc, y_mean = yc,\
                                  x_stddev=narrow_sigma_x, y_stddev=narrow_sigma_y, theta = narrow_theta)
    wide_psf = models.Gaussian2D(amplitude = wide_amplitude, x_mean=xc+dx, y_mean=yc+dy,\
                                  x_stddev=wide_sigma_x, y_stddev=wide_sigma_y, theta=wide_theta)
    psf = wide_psf(x,y)+narrow_psf(x,y)
    return psf


def build_analytical_model(p):
    ''' Takes in model parameters to return a 2d model image:
            p = xca,yca,xcb,ycb,xcc,ycc,dx,dy,ampa,ampb,ampc,ampratio,bkgdfill,sigmax,sigmay,sigmax2,sigmay2,theta,theta2 
                (1 = narrow Gaussian, 2 = wide Gaussian; a = central star, b/c = companions)
        Input: 
            p: model parameters (1d array)
        Returns: 
            model: 2d model image
    '''
    psfa = build_2d_gaussian(p[0],p[1],p[6],p[7],p[8],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18])
    psfb = build_2d_gaussian(p[2],p[3],p[6],p[7],p[9],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18])
    psfc = build_2d_gaussian(p[4],p[5],p[6],p[7],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18])

    # Build the background level:
    bkgd = np.ndarray(shape=(xsize,ysize), dtype=float)
    bkgd.fill(p[12])

    # Assemble the model image:
    psf = psfa + psfb + psfc + bkgd
    return psf

def chi_squared(data,model,error):
    chisquared_per_pixel = ((data-model)/error)**2
    chisquared = np.sum(chisquared_per_pixel)
    return chisquared

def accept_reject(chisquare_current,chisquare_proposal):
    # Determine acceptance probability:
    p_accept = np.exp(-(chisquare_proposal - chisquare_current)/2.)
    # Determine random 'dice roll':
    dice = np.random.rand()
    if dice < p_accept:
        accept = 'yes'
    else:
        accept = 'no'
    return accept, p_accept, dice


############################### Observational data: ##############################

######## Read in image:
parser = argparse.ArgumentParser()
parser.add_argument("image",type=str)
args = parser.parse_args()

image = fits.open(args.image)[0].data
imhdr = getheader(args.image)

# Get directory to store output:
d = args.image.split('/')
directory = ''
for i in range(len(d)-1):
    directory = directory + str(d[i]) + '/'
    
# Place the output in a directory named imagenumber_apf_results:
output_directory =  directory + d[-1].split('.')[-3] + '_apf_results/'
print output_directory
if rank == 0:
    os.system('mkdir '+ output_directory)

######## Mask saturated pixels so they do not contribute to the fit:
itime = float(imhdr['itime'])*1000.
coadds = float(imhdr['coadds'])
multisam = float(imhdr['multisam'])
sampmode = imhdr['sampmode']

# Determine saturation level:
if sampmode == 3:
    satlevel = coadds * 24000.0 * (1.0 - 0.1*(multisam-1.0)/(itime/1000.))
else:
    satlevel = coadds * 22000.0

# Make a new copy of the image where pixels at 80% of saturation are masked out:
image_nanmask = np.ma.masked_greater(image, 0.8*satlevel)
if rank==0:
    print 'Max pixel value in image:',np.max(image)
    print 'Masking pixels greater than ',0.8*satlevel
    print 'I have masked',np.product(np.shape(image))-image_nanmask.count(),'pixels'

########## Calculate the error in each pixel:
# Readnoise error:
# The readnoise error is calculated from the readnoise characterization in the NIRC2 manual.
rnoise = np.ndarray(shape=image.shape, dtype=float) #create the array of the same shape as the image
if sampmode == 3.0:
    readnoise = (38.0/np.sqrt(multisam)) * (np.sqrt(coadds))
elif sampmode == 2.0:
    readnoise = 38 * (np.sqrt(coadds))
else: 
    readnoise = 38 * (np.sqrt(coadds))
rnoise.fill(readnoise)

# Poisson error:
pois = np.sqrt(np.abs(image))

#error: an array of the same size as the image containing the value of the error in each pixel
err = np.sqrt(rnoise**2+pois**2)

################################ Set up some things: ###########################################

# Set jump widths:
# (empirically determined)
#positionswidth = 0.01 #pixels
#positioncwidth = 0.3 #pixels
#offsetxwidth = 0.08 #pixels
#offsetywidth = 0.09 #pixels
#ampswidth = 0.0025 #log counts
#ampcwidth = 0.02 #log counts
#ampratiowidth = 0.001 #fraction of wide gaussian to narrow gaussian amplitude
#bkgdwidth = 0.0008 #log counts
#sigmawidth = 0.002 #log pixels
#sigma2width = 0.001 #log pixels
#thetawidth = 0.008 #radians
#theta2width = 0.01 #radians

widths = np.array([0.01,
                       0.01,
                       0.3,
                       0.3,
                       0.3,
                       0.3,
                       0.08,
                       0.09,
                       0.0025,
                       0.02,
                       0.02,
                       0.001,
                       0.0008,
                       0.002,
                       0.002,
                       0.001,
                       0.001,
                       0.008,
                       0.01])

# Set model size to same size as the image:
ysize,xsize = image.shape[1],image.shape[0]


################################### Initialize everything: #################################################
# NIRC2 specific parameters:
FWHM=50 #mas
Pixscale = 9.95 #milliarcsec/pixel
FWHM = FWHM / Pixscale
sigma = FWHM/2.35

##### Pull in output from step 1 and build initial parameters array:
fileguess = directory + d[-1].split('.')[-3] + '_initialguess'
guess = np.loadtxt(open(fileguess,"rb"),delimiter=' ')
# Get position of star/companion from step 1 output:
xca,yca,xcb,ycb,xcc,ycc = guess[0],guess[1],guess[2],guess[3],guess[4],guess[5]

# Use image values at those points as initial amplitude guess:
ampa = image[int(yca-0.5+1),int(xca-0.5+1)]
ampb = image[int(ycb-0.5+1),int(xcb-0.5+1)]
ampc = image[int(ycc-0.5+1),int(xcc-0.5+1)]
# Get the median noise level in a region of the image as the bkgd level:
box = image[int(guess[7]):int(guess[7])+10, int(guess[6]):int(guess[6])+10]
bkgd = np.median(box)

parameters = np.array([xca,yca,xcb,ycb,xcc,ycc,0.,0.,ampa,ampb,ampc,0.2,bkgd,sigma,sigma,sigma*3,sigma*3,0.,0.,0.])
'''
Parameters indicies:
0 xca
1 yca
2 xcb
3 ycb
4 xcc
5 ycc
6 dx
7 dy
8 ampa
9 ampb
10 ampc
11 ampratio
12 bkgd
13 sigmax
14 sigmay
15 sigmax2
16 sigmay2
17 theta
18 theta2
19 chisquared
'''

# These parameter indicies need normal priors:
# the positions, dx,dy,ampratio,thetas
norm = [0,1,2,3,4,5,6,7,11,17,18]
# These parameter indicies need log normal priors:
# the amps, bkgd,sigmas
lognorm = [8,9,10,12,13,14,15,16]

# Initialize parameter tracking arrays:
total_tries,total_accept = np.zeros(parameters.shape[0]-1),np.zeros(parameters.shape[0]-1)
# Initialize the total parameters tracking array:

total_parameters = [[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],\
                        [np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan]]

# (Because of how np.hstack works, each element in the array must be its own list, sadly.)

# Build initial model:
model = build_analytical_model(parameters)
# Get initial chisquared value:
chi = chi_squared(image_nanmask,model,err)
if rank == 0:
    print 'Found initial chi-squared:',chi
# Enter the initial chi squared into the parameters array:
parameters[-1] = chi

if rank == 0:
    print 'Initial guess:',parameters


################################### Begin loop ########################################################
if rank==0:
    print
    print('Beginning loop...')
    print 'Burn in = ',burn_in
count = 0
# Run the loop until each parameter has been tried a minimum number of times:
while np.min(total_tries) < accept_min:
    # Select random parameter to vary:
    rand = np.random.randint(0,parameters.shape[0]-1)
    # Iterate the trials tracker:
    total_tries[rand] += 1
    # Get a new value for that parameter:
    if rand in norm:
        new = proposal(parameters[rand],widths[rand],1)[0]
    elif rand in lognorm:
        new = logproposal(parameters[rand],widths[rand],1)[0]

    #print 'I will be testing index', rand
    #print 'The old value is',parameters[rand]
    #print 'using jump width ', widths[rand]
    #print 'The new value is',new

    #Build a new model:
    parameters_proposal = parameters.copy()
    parameters_proposal[rand] = new
    model_proposal = build_analytical_model(parameters_proposal)
    # Get a new chi-squared value:
    chi_proposal = chi_squared(image_nanmask,model_proposal,err)
    #print 'old chisquared:',parameters[-1]
    #print 'new chisquared:',chi_proposal
    # Determine if it should be accepted:
    accept, p_accept, dice = accept_reject(parameters[-1],chi_proposal)
    #print 'Was it accepted?', accept
    #print 'the probability was',p_accept,'and the dice roll was:',dice

    # If accepted:
    if accept == 'yes':
        # Iterate the acceptance tracker:
        total_accept[rand] +=1
        # Place the new value into the parameters array:
        parameters[rand] = new
        # Place the new chisquared value into the parameters array:
        parameters[-1] = chi_proposal
    else:
        # If not accepted, keep the old parameter value
        # and chisquared value in the parameters array
        pass

    count += 1

    #print 'Current state of parameters array:',parameters

    # wait for all processes to check in so they stay in sync:
    comm.barrier()

    # Throw away the first X number of steps, to constitute burn in.  Start writing out each
    # walker's status after burn in is accomplished:
    if count >= burn_in:
        if count == burn_in and rank == 0:
            print 'Burn in accomplished, commence writing out progress.'
        # Stack the current parameters status array onto the previous arrays:
        stack_parameters = [[parameters[0]],[parameters[1]],[parameters[2]],[parameters[3]],\
                            [parameters[4]],[parameters[5]],[parameters[6]],[parameters[7]],[parameters[8]]\
                            ,[parameters[9]],[parameters[10]],[parameters[11]],[parameters[12]],[parameters[13]]\
                            ,[parameters[14]],[parameters[15]],[parameters[16]],[parameters[17]],[parameters[18]],[parameters[19]]]
                            # (Because of how np.hstack works, each element in the array must be its own list, sadly)
        total_parameters = np.hstack([total_parameters,stack_parameters])

        # Every so many loops, write out the state of the total parameters array:
        # (Overwrites the old file each time)
        if np.sum(total_tries) % 10 == 0:
            # Write chains to file:
            b = open(output_directory + str(rank)+'_finalarray_mpi.csv', 'w')
            a = csv.writer(b)
            a.writerows(np.transpose(total_parameters))
            b.close()
            # Write acceptance rate:
            z = open(output_directory + str(rank)+'_acceptance_rate.csv', 'w')
            acceptance_rate = total_accept / total_tries
            z.write(str(acceptance_rate))
            z.close()
            
    if np.sum(total_tries) % 10 == 0 and rank == 0:
        print 'Loop count:',np.sum(total_tries)
    if np.sum(total_tries) % 100 == 0 and rank == 0:
        print 'Acceptance rate:',total_accept / total_tries


print 'Rank ',rank, 'done with loop'
print
print 'The output for xcs looks like:'
c = np.genfromtxt(output_directory+str(rank)+'_finalarray_mpi.csv',delimiter=',')
print c[:,0]



