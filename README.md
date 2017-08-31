# olpefit
OLPE Fit (Orbits of Long Period Exoplanets) is an MCMC fitting algorithm for finding the position of a star and a companion in pixel space on Keck/NIRC2 infrared array images.  It uses astropy 2d Gaussian modeling to generate a model of the object's PSF using two Gaussians - a narrow one for the diffraction limited core and a wider one for the Airy rings.  It uses a Gibbs sampler to sample 14 dimension parameter space of the model: 4 position parameters (x and y pixel positions for the star and the companion), amplitude of star and companion Gaussians, the background noise level, the ratio between the max peak of the narrow and wide Gaussians, the std deviation of both Gaussians in the x and y direction, and the rotation angle of the major axis of the two Gaussians.  It creates an initial model based on the values in the original image, computes the goodness of fit for the model, then steps through varying each parameters by a small amount and tests for a better fit.  It walks around this 14 dim parameter space until it has found minima for (almost) all the parameters.  It does not look at standard deviations or thetas for convergence.
OLPE Fit can accept any number of inidividual walkers exploring parameter space individually.  It looks for convergence by measuring the standard deviation between the mean values of each walker, and the scatter among data points inside the walkers.  When the std dev between the means has fallen below a set fraction of the overall scatter for a parameter, OLPE Fit declares that that parameter has converged.  When all parameters (except the sigmas and thetas) have converged, OLPE Fit finishes.
After running OLPE Fit, you will have a new directory named imagenumber_olpefit_results, within which are .csv files for each parameter array.  The array will be NWalkers x number of trials for that parameter.

Variables in the script which can be set:
NWalkers = number of walkers used
accept_min = Sets the burn-in rate.  Jumps will not be recorded or monitored until a parameters has made a number of jumps larger than accept_min
convergenceratio = Percent required for scatter between means relating to overall scatter to decide convergence.  If convergenceratio = 0.1 then the scatter between the means must fall below 10% of the overall scatter to be considered convereged.
(parameter)width = Sets the jump width for each parameter - the standard deviation for the normal distribution from which the new parameter values are drawn.

Specify the image to be analyzed when running OLPE Fit from the terminal.  ex: $python olpe_fit.py path_to_image/image


# olpefit_pasep
This script takes the array outputs of olpefit and determines the position angle and separation of the companion from the host star.  It truncates the parameter arrays to the length of the shortest array (to allow array math on arrays of the same shape).  It applies the NIRC2 distortion solution to the x and y position arrays by first asking you to input if the image was obtained before April 13th 2015 or after (the date of NIRC2 realignment), then applies the appropriate distortion solution.  It uses the dedistorted pixel positions to computer separation and position angle for each position, generating a new separation and position angle array.  It then compensates for telescope rotation and motion during the observation (if the telescope was in vertical angle mode for the observation).  It then computes the mean separation and position angle and standard deviation, generating an uncertainty in both parameters for each individual image analyzed.  It writes these values into the text file log for the entire observation epoch.  Lastly, the script creates a triangle plot for all 14 parameters from OLPE Fit and puts in the same directory as the OLPE Fit output arrays.

# olpefit_tacc
These scripts accomplish the same as olpe fit but are engineered to run more effectively on TACC supercomputer. olpefit_tacc_interactivepart contains the interactive portion of the model generation phase of OLPE Fit so it can be run quickly, and its outputs feed into olpefit_tacc allowing that to be run in a "fix it and forget it" way.  This is so that all the interactive part can be done at once for a whole epoch prior to running an sbatch on TACC.

## NIRC2 bad pix
Bad pix is a list of known bad pixels of NIRC2.  OLPE Fit flags these pixels and marks then as NaNs in image data so they don't contribute to the fitting algorithm.

## Disortion Solution Files
nirc2_distort_X_post20150413_v1.fits and nirc2_distort_Y_post20150413_v1.fits are look up tables for the NIRC2 distortion solution by Service et.al. 2016 (http://iopscience.iop.org/article/10.1088/1538-3873/128/967/095004/pdf)
nirc2_X_distortion.fits and nirc2_Y_distortion.fits are distortion solutions by Yelda et.al. 2010 (http://iopscience.iop.org/article/10.1088/0004-637X/725/1/331/pdf)
