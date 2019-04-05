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
#                      Supplemental Definitions                                #
################################################################################
# Definitions of functions for performing the photometry and astrometry in step 4 of GRAND

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def dist_circle(xcen,ycen,x,y):
    dx = np.abs(x-xcen)
    dy = np.abs(y-ycen)
    dist = np.sqrt(dx**2+dy**2)
    return dist

def aperture_annulus(sx,sy,r,r_in,r_out):
    """
    Returns a list of indicies which fall within the specified circular aperture and
    annulus around the source
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
        
    Return:
        aperture_indicies (np.array): list of indicies within the aperture, in x,y
        annulus_indicies (np.array): list of indicies within the annulus, in x,y
    Written by: Logan Pearce, 2018
    """
    import warnings
    warnings.filterwarnings('ignore')
    # Measure the distances of all pixels in the image to the center of the source:
    xarray = np.int_(np.linspace(0,image.shape[1],image.shape[1]))
    yarray = np.int_(np.linspace(0,image.shape[0],image.shape[0]))
    index = np.linspace(0,xarray.shape[0]-1,xarray.shape[0])
    distances = np.zeros((xarray.shape[0],yarray.shape[0]))
    for xi,i1 in zip(xarray,index):
        for yi,i2 in zip(yarray,index):
            distances[i1,i2] = dist_circle(np.int_(sx),np.int_(sy),xi,yi)
    distances = np.int_(distances)
    # Make an array of indicies which fall within a annulus of specified inner and outer radius:
    annulus_indicies = np.where((distances>=np.int_(r_in))&(distances<=np.int_(r_out)))
    aperture_indicies = np.where(distances<=r)
    return aperture_indicies,annulus_indicies

def signal_noise_ratio(image,sx,sy,r,r_in,r_out):
    """
    Returns signal to noise ratio, signal value, noise value
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
    Return:
        snr (float): signal-to-noise ratio
        signal (float): sum of all pixel values within the specified aperture minus sky background
        noise (float): noise in the signal calculated as std deviation of pixels in the sky annulus times sqrt of the area of the
        signal aperture
        poisson_noise (float): snr determined from the poisson noise in the source signal.  Poisson noise = sqrt(counts) [because variance
        of a poisson distribution is the parameter itself].  Poisson snr = Signal/sqrt(signal) = sqrt(signal)
    Written by: Logan Pearce, 2018
    """
    import warnings
    warnings.filterwarnings('ignore')
    ap_an = aperture_annulus(sx,sy,r,r_in,r_out)
    ap,skyan = ap_an[0],ap_an[1]
    apsum = np.sum(image[ap[1],ap[0]])
    skysum = np.sum(image[skyan[1],skyan[0]])
    skyarea = np.shape(skyan)[1]
    averagesky = skysum/skyarea
    signal = (apsum - np.shape(ap)[1]*averagesky)
    poisson_noise = np.sqrt(signal)
    noise = np.std(image[skyan[1],skyan[0]])
    noise = noise*np.sqrt(np.shape(ap)[1])
    snr = signal/noise
    return snr,signal,noise,poisson_noise

def snr_astropy(image,sx,sy,r,r_in,r_out):
    """
    Returns signal to noise ratio, signal value, noise value using Astropy's Photutils module
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
    Return:
        snr (float): signal-to-noise ratio
        signal (float): sum of all pixel values within the specified aperture minus sky background
        noise (float): noise in the signal calculated as std deviation of pixels in the sky annulus times sqrt of the area of the
        signal aperture
        poisson_noise (float): snr determined from the poisson noise in the source signal.  Poisson noise = sqrt(counts) [because variance
        of a poisson distribution is the parameter itself].  Poisson snr = Signal/sqrt(signal) = sqrt(signal)
    Written by: Logan Pearce, 2018
    """
    import warnings
    warnings.filterwarnings('ignore')
    from photutils import CircularAperture, CircularAnnulus
    positions = (cx-1,cy-1)
    ap = CircularAperture(positions,r=r)
    skyan = CircularAnnulus(positions,r_in=11,r_out=14)
    apsum = ap.do_photometry(image)[0]
    skysum = skyan.do_photometry(image)[0]
    averagesky = skysum/skyan.area()
    signal = (apsum - ap.area()*averagesky)[0]
    n = ap.area()
    ap_an = aperture_annulus(sx,sy,r,r_in,r_out)
    skyan = ap_an[1]
    poisson_noise = np.sqrt(signal)
    noise = noise = np.std(image[skyan[1],skyan[0]])
    noise = noise*np.sqrt(n)
    snr = signal/noise
    return snr,signal,noise,poisson_noise
