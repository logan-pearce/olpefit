# olpefit
OLPE Fit (Orbits of Long Period Exoplanets) is an MCMC fitting algorithm for finding the position of a star and a companion in pixel space on Keck/NIRC2 infrared array images.  It uses astropy 2d Gaussian modeling to generate a model of the object's PSF using two Gaussians - a narrow one for the diffraction limited core and a wider one for the Airy rings.  It uses a Gibbs sampler to sample 14 dimension parameter space of the model: 4 position parameters (x and y pixel positions for the star and the companion), amplitude of star and companion Gaussians, the background noise level, the ratio between the max peak of the narrow and wide Gaussians, the std deviation of both Gaussians in the x and y direction, and the rotation angle of the major axis of the two Gaussians.  It creates an initial model based on the values in the original image, computes the goodness of fit for the model, then steps through varying each parameters by a small amount and tests for a better fit.  It walks around this 14 dim parameter space until it has found minima for (almost) all the parameters.  It does not look at standard deviations or thetas for convergence.
OLPE Fit can accept any number of inidividual walkers exploring parameter space individually.  It looks for convergence by measuring the standard deviation between the mean values of each walker, and the scatter among data points inside the walkers.  When the std dev between the means has fallen below a set fraction of the overall scatter for a parameter, OLPE Fit declares that that parameter has converged.  When all parameters (except the sigmas and thetas) have converged, OLPE Fit finishes.

Variables in the script which can be set:
NWalkers = number of walkers used
accept_min = Sets the burn-in rate.  Jumps will not be recorded or monitored until a parameters has made a number of jumps larger than accept_min
convergenceratio = Percent required for scatter between means relating to overall scatter to decide convergence.  If convergenceratio = 0.1 then the scatter between the means must fall below 10% of the overall scatter to be considered convereged.
(parameter)width = Sets the jump width for each parameter - the standard deviation for the normal distribution from which the new parameter values are drawn.

Specify the image to be analyzed when running OLPE Fit from the terminal.  ex: $python olpe_fit.py path_to_image/image
