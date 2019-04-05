# LAPF:
### Logan's Analytical PSF Fitter
    Fits a Gaussian 2d PSF model to NIRC2 data for companion to a central (unobscured) star
    using a Gibbs sampler Metroplis-Hastings MCMC.  Runs in parallel, where each process acts an 
    independent walker.  For details, see Pearce et. al. 2019.
       Step 1: Locate central object, companion, and empty sky area in the image
       Step 2: MCMC iterates on parameters of 2D model until a minimum number of trials
               are conducted on each parameter.  Each process outputs their chain to an 
               individual file
       Step 3: Take in output of step 2 and apply corrections to determine relative separation
               and position angle and corresponding metrics.
