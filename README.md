# LAPF:
### Logan's Analytical PSF Fitter

Fits a Gaussian 2d PSF model to Keck/NIRC2 data for companion to a central (unobscured) star
using a Gibbs sampler Metroplis-Hastings MCMC.  Runs in parallel, where each process acts an 
independent walker.  For details, see Pearce et. al. 2019.

LAPF works in three steps: <br>
       Step 1: Locate central object, companion, and empty sky area in the image <br>
       Step 2: MCMC iterates on parameters of 2D model until a minimum number of trials <br>
               are conducted on each parameter.  Each process outputs their chain to an <br>
               individual file. <br>
               Designed to run on the Texas Advance Computing Center Lonestar 5 compter in parallel processing <br>
       Step 3: Take in output of step 2 and apply corrections to determine relative separation <br>
               and position angle and corresponding metrics. <br>
