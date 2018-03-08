# Step 4 for MPI runs of Step 2
# This is part of the OLPE Fit pipeline.  It takes in the parameters results of
# the mcmc fitter (step 2) and determines relative separation of companion from host in millarcseconds and position angle
# of companion relative to North.  It applies the appropriate distortion solution corrections (Yelda 2010 for
# observations before April 2015 and Service 2016 for observations after April 2015, due to NIRC2 camera realignment).
# It corrects telescope offsets, and applies atmospheric and chromatic distortion corrections.
# For atmospheric corrections, it draws condition information from the header of the image, and uses astropy to retrieve
# the star's RA/Dec information from Sesame and convert to Alt/Az, and to retrieve the location of Keck observatory.  So it requires
# astropy to be installed and an internet connection to execute.
# (No, there is no step 3.  There used to be, but it is now obsolete)
#
# Inputs:
#    - .csv files from step 2 output (located in the output folder from Step 2)
#    - Distortion solution look up tables from Yelda 2010 and Service 2016 (located in the same directory as this script)
#    - Image header information
# Outputs:
#    - A text file called "epoch_positions_olpefit_pasep" which lists each image's sep, PA, and std dev on each, S/N ratio for
#        star and companion, and FWHM for input into position agrigator.
#    - A text file called "epoch_positions_olpefit_log" which contains more detail about each image results
#    - A corner plot of all parameters in the fit for each image
#
#
# From the terminal, execute as follows:
# python olpe_fit_tacc_step4_mpi.py path_to_image_file
#
# Written by Logan A. Pearce


import numpy as np
import argparse
from astropy.io import fits
from astropy.io.fits import getheader
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u

#######################################################################################################
######################################### Definitions #################################################
#######################################################################################################

def zenith_correction_factor(zo):
    fz_dict = {'0':0.,'10':0.,'20':0.,'30':0.,'35':0.,'40':2e-4,'45':6e-4,'50':12e-4,\
         '55':21e-4,'60':34e-4,'65':56e-4,'70':97e-4} #create a dictionary for the lookup table
    gz_dict = {'0':4e-4,'10':4e-4,'20':4e-4,'30':5e-4,'35':5e-4,'40':5e-4,'45':6e-4,'50':6e-4,\
         '55':7e-4,'60':8e-4,'65':10e-4,'70':13e-4}
    if zo >= 0 and zo <= 30:
        z = str(int(np.round(zo,decimals=-1))) #round to nearest 10s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    elif zo > 30 and zo <= 70:
        z = str(int(np.round(zo/5.0)*5.0)) #round to nearest 5s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    else:
        return 'Atmospheric correction not required, zenith greater than 70 deg'


def atm_corr(zo,p,t,lamb,hum):
    fz,gz = zenith_correction_factor(zo)
    zo = np.radians(zo)
    po = 101325. #Pa
    Ro = 60.236*np.tan(zo)-0.0675*(np.tan(zo)**3) # In arcseconds
    # Atm correction:
    F = (1.-(0.003592*(t-15.))-(5.5e-6*((t-7.5)**2)))*(1.+fz)
    G = (1.+(0.943e-5*(p-po))-((0.78e-10)*((p-po)**2)))*(1.+gz)
    R = Ro*(p/po)*(1.055216/(1.+0.00368084*t))*F*G
    # Chromatic effects:
    R = R*(0.98282+(0.005981/(lamb**2)))
    # Water vapor correction:
    f = hum/100. #convert percent to decimal
    # Calculate saturation vapor pressure:
    # Using Buck eqn (https://en.wikipedia.org/wiki/Vapour_pressure_of_water):
    Psat = 0.61121*np.exp((18.678-(t/234.5))*(t/(257.14+t))) # in kPa
    Psat = Psat*1000 #convert to Pa
    # Calculate water vapor partial pressure: (http://www.engineeringtoolbox.com/relative-humidity-air-d_687.html)
    Pw = f*Psat
    R = R*(1-0.152e-5*Pw - 0.55e-9*(Pw**2)) # R is in arcseconds
    zo = np.degrees(zo)
    z = zo+(R/3600.)  # Convert R to degrees
    return R/3600.,z


lamb_dict = {'z':1.0311,'Y':1.0180,'J':1.248,'H':1.633,'K':2.196,'Ks':2.146,'Kp':2.124,\
             'Lw':3.5197,'Lp':3.776,'Ms':4.670} #Dictionary of median wavelength for NIRC2 filters in micrometers


#######################################################################################################
#################################### User defined settings: ###########################################
#######################################################################################################
#                                                                                                     #
#       Change the value of these variables to desired settings:                                      #
#           -additional_burnin: number of data points discarded before computing average value        #
#                Change this so that the only data points remaining are the ones in the minima        #
#                                                                                                     #
#######################################################################################################

additional_burnin = 22000

#######################################################################################################
############################################# Import data #############################################
#######################################################################################################
             
# Get the file name from the entered argument:
parser = argparse.ArgumentParser()
parser.add_argument("image_filename",type=str)
args = parser.parse_args()
filename=args.image_filename

image1 = fits.open(args.image_filename)
image = image1[0].data
imhdr = getheader(args.image_filename)

epoch = filename.split('/')[1]
epoch = int(epoch.split('_')[0])

if epoch >= 2015:
    prepost = "post"
else:
    prepost = "pre"

directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_mpi'

print 'Burn in:',additional_burnin

#######################################################################################################
########################################## Import arrays ##############################################
#######################################################################################################

print 'Importing parameter arrays...'
c = np.genfromtxt(directory+'/0_finalarray_mpi.csv',delimiter=',') # This just measures the size of the
# outputted parameter arrays
length = c.shape[1]
ncor = 48
index = ncor
xcs,ycs,xcc,ycc = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
amps,ampc,ampratio,bkgd = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chisquare = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),\
    np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
# Place each walker's parameter into a collated parameter array - the columns of the array are each walker
for i in range(index):
    a = np.genfromtxt(directory+'/'+str(i)+'_finalarray_mpi.csv',delimiter=',')
    xcs[:,i],ycs[:,i],xcc[:,i],ycc[:,i] = a[0],a[1],a[2],a[3]
    amps[:,i],ampc[:,i],ampratio[:,i],bkgd[:,i] = a[4],a[5],a[6],a[7]
    sigmax[:,i],sigmay[:,i],sigmax2[:,i],sigmay2[:,i],theta[:,i],theta2[:,i],chisquare[:,i] = a[8],a[9],a[10],a[11],a[12],a[13],a[14]

#Determine the number of walkers used in the fit:
NWalkers = ncor

# Give the parameters some additional burn in:
xcs = xcs[additional_burnin:length,:] 
ycs = ycs[additional_burnin:length,:]
xcc = xcc[additional_burnin:length,:] 
ycc = ycc[additional_burnin:length,:]
amps = amps[additional_burnin:length,:]
ampc = ampc[additional_burnin:length,:]
ampratio = ampratio[additional_burnin:length,:]
bkgd = bkgd[additional_burnin:length,:]
sigmax = sigmax[additional_burnin:length,:]
sigmay = sigmay[additional_burnin:length,:]
sigmax2 = sigmax2[additional_burnin:length,:]
sigmay2 = sigmay2[additional_burnin:length,:]
theta = theta[additional_burnin:length,:]
theta2 = theta2[additional_burnin:length,:]
print "Number of total jumps per walker:",length
print "Number of total samples after burn-in: ",xcs.shape[0]*ncor

# Add one pixel to each value because python indexes starting at zero, and fits files start at 1 (for the distortion lookup table):
xcs = xcs+1 
ycs = ycs+1
xcc = xcc+1
ycc = ycc+1

#######################################################################################################
####################################### Correct plate distortion ######################################
#######################################################################################################
print 'Correcting plate scale distortion...'
if prepost == 'pre' or prepost =='Pre':
    # Open the lookup tables of Yelda 2010:
    x_dist = fits.open('nirc2_X_distortion.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('nirc2_Y_distortion.fits')
    y_dist = y_dist[0].data
    pixscale = 9.952 #In mas
elif prepost == 'post' or prepost =='Post':
    # Open the lookup tables of Service 2016:
    x_dist = fits.open('nirc2_distort_X_post20150413_v1.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('nirc2_distort_Y_post20150413_v1.fits')
    y_dist = y_dist[0].data
    pixscale = 9.971 #In mas
        
# Convert pixel locations to integers to feed into lookup table:
xcc_int = np.int_(xcc)
ycc_int = np.int_(ycc)
xcs_int = np.int_(xcs)
ycs_int = np.int_(ycs)

# If the image is sized differently than 1024x1024, adjust the lookup indices to grab the correct distortion correction
#value:
xdiff=int(np.ceil(0.5*(1024-image.shape[1])))
ydiff=int(np.ceil(0.5*(1024-image.shape[0])))

xcc_int = xcc_int+xdiff
xcs_int = xcs_int+xdiff
ycc_int = ycc_int+ydiff
ycs_int = ycs_int+ydiff

# Add the distortion solution correction to each datapoint in the position arrays:
xcc_dedistort = xcc + x_dist[ycc_int,xcc_int]
ycc_dedistort = ycc + y_dist[ycc_int,xcc_int]
xcs_dedistort = xcs + x_dist[ycs_int,xcs_int]
ycs_dedistort = ycs + y_dist[ycs_int,xcs_int]

dy = ycc_dedistort-ycs_dedistort
dx = xcc_dedistort-xcs_dedistort

# Compute relative RA/Dec with star at 0,0 in pixel space:
# Convert to RA/Dec in milliarcseconds:
RA = dx*pixscale #Neg because RA is defined increasing right to left
Dec = dy*pixscale

RA,RAstd = -np.mean(RA),np.std(RA)
Dec,Decstd = np.mean(Dec),np.std(Dec)

#######################################################################################################
######################################## Compute separation ###########################################
#######################################################################################################
print 'Computing separation and position angle...'
rsquare = (dy**2)+(dx**2)
r = np.sqrt(rsquare)
# ^ separation in pixel space
sep = r*pixscale # Convert to mas

#######################################################################################################
####################################### Compute position angle ########################################
#######################################################################################################

pa = np.arctan2(-dx,dy)
pa = np.degrees(pa)

# Rotation angle correction:
p = imhdr['PARANG']
r = imhdr['ROTPPOSN']
i = imhdr['INSTANGL']
e = imhdr['EL']

if prepost == 'pre' or prepost =='Pre':
    rotation_angle = p+r-e-i-0.252  # Yelda 2010 solution
elif prepost == 'post' or prepost =='Post':
    rotation_angle = p+r-e-i-0.262 # Service 2016 solutionn

pa = pa+rotation_angle

# Vertical angle mode correction:
mode = imhdr['ROTMODE']

if mode != 'vertical angle':
    #vertical angle rotation compensation not required
    a = str('no')
    pass
else:
    #Calculate exposure total length:
    a=str('yes')
    t1 = imhdr['EXPSTART']
    t2 = imhdr['EXPSTOP']
    def get_sec(time_str):  #Converts H:M:S notation into seconds
        h, m, s = time_str.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)
    t1 = get_sec(t1)
    t2 = get_sec(t2)
    dt = t2-t1
    #Calculate d(eta)/dt:
    #(source: http://www.mmto.org/MMTpapers/pdfs/itm/itm04-1.pdf)
    az = imhdr['AZ']
    az = az*(np.pi/180) #Convert to radians
    el = imhdr['EL']
    el = el*(np.pi/180)
    L  = 0.346036 #Latitude of Keck
    etadot = (-0.262) * np.cos(L) * np.cos(az) / np.cos(el) #Change in angle in rad/hr
    #convert to degrees per second:
    etadot = etadot*(180/np.pi)/3600
    rotcorr = etadot*(dt/2)
    
    #add to theta:
    pa = pa+rotcorr

# If the computed position angle is negative, add 360 deg:
if np.mean(pa) < 0:
    pa = pa+360.
else:
    pass

##################################################################################################################
####################################### Atmospheric distortion correction ########################################
##################################################################################################################
print 'Correcting atmospheric distortion...'
# Get zenith angle of star:
# Get Keck location:
keck = EarthLocation.of_site('Keck Observatory')
# Get RA/Dec of host star using Sesame:
object = filename.split('/')[0]
gsc = SkyCoord.from_name(object)
# Get observation time and date from header:
obstime = str(imhdr['UTC'])
obsdate = str(imhdr['DATE-OBS'])
obs = obsdate + ' ' + obstime
obstime = Time(obs) # Make it an astropy time object
# Convert to alt/az:
gscaltaz = gsc.transform_to(AltAz(obstime=obstime,location=keck))
star_alt = gscaltaz.alt.deg
# Convert into apparent zenith angle in degrees:
zos = 90.-star_alt

# Determine companion RA/Dec and convert to Alt/Az:
delta_dec = np.mean(ycs_dedistort) - np.mean(ycc_dedistort) # Change in dec in pixel space
delta_ra = np.mean(xcs_dedistort) - np.mean(xcc_dedistort) # change in RA in pixel space
delta_dec = delta_dec*(pixscale/1000)/3600. #<-convert to degrees
delta_ra = delta_ra*(pixscale/1000)/3600.
# Compute companion ra/dec from star's sesame ra/dec:
comp_dec = gsc.dec.deg - delta_dec
comp_ra = gsc.ra.deg + delta_ra
# Make a skycoords object (RA/Dec):
gscb = SkyCoord(ra=comp_ra*u.degree, dec=comp_dec*u.degree, frame='icrs')
# Convert to alt/az:
gscbaltaz = gscb.transform_to(AltAz(obstime=obstime,location=keck))
comp_alt = gscbaltaz.alt.deg
# Convert into apparent zenith angle in degrees:
zoc = 90.-comp_alt

###### Get atmospheric correction factors:
band = imhdr['FWINAME']
lamb = float(imhdr['WAVECNTR'])
p,T,hum = float(imhdr['WXPRESS']),float(imhdr['WXOUTTMP']),float(imhdr['WXOUTHUM'])
p = p*100 # 1 mbar = 100 Pa; should be less than 101325 Pa because of altitude

# Get distortion correction factor for star and companion:
Rs,zs = atm_corr(zos,p,T,lamb,hum)
Rc,zc = atm_corr(zoc,p,T,lamb,hum)
# Compute the amount in mas that the companion is shifted more than the star:
deltaR = (Rc-Rs)*3600*1000 
# Get telescope zenith angle (rotation from north):
parantel = imhdr['PARANTEL']

# Apply correction factor to sep/PA:
# If the companion is below the star (relative to north) and above the zenith line
# (90 < PA-parantel < 270) sep and PA should increase. If the companion is below the star
# and below the zenith line, sep should increase and PA decrease.  Above the star and zenith
# line, sep increase and PA decrease.  Above the star and below the zenith line both should increase.

pop = np.mean(pa)-parantel
if pop >= 90. and pop <= 270. : # The companion is "below" the star relative to north
    # Apply to separation:
    deltasep = deltaR * np.cos(np.radians(180. - (pa - parantel)))
    sep_corrected = sep + deltasep
    # Apply to angular distance:
    pa_mas = (sep)*np.tan(np.radians(pa))
    deltapa = deltaR * np.sin(np.radians(180. - (pa - parantel)))
    pa_mas_corrected = pa_mas + deltapa
    # Convert back into angle:
    pa_angle = np.arctan(pa_mas_corrected/sep_corrected)
    if np.mean(pa) >= 90.:
        pa_angle = np.degrees(pa_angle)+180.
    else:
        pa_angle = np.degrees(pa_angle)
else:  # The companion is "above" the star
    # Apply to separation:
    deltasep = deltaR * np.cos(np.radians(pa - parantel))
    sep_corrected = sep + deltasep
    # Apply to angular distance:
    pa_mas = (sep)*np.tan(np.radians(pa))
    deltapa = deltaR * np.sin(np.radians(pa - parantel))
    # Convert back into angle:
    pa_mas_corrected = pa_mas + deltapa
    if pop > 270. and pop > 270:
        pa_angle = np.arctan(pa_mas_corrected/sep_corrected)+360.
    elif pop > 270. and np.mean(pa) < 270:
        pa_angle = np.arctan(pa_mas_corrected/sep_corrected)+180.
    else:
        pa_angle = np.arctan(pa_mas_corrected/sep_corrected)

##################################################################################################################
####################################### Compute mean sep and pa value ############################################
##################################################################################################################
print 'Computing median values:'
sep_mean,sep_stdev = np.median(sep_corrected),np.std(sep_corrected)
pa_mean,pa_stdev=np.median(pa_angle),np.std(pa_angle)

print "r = ", sep_mean, "pa = ", pa_mean

##################################################################################################################
########################################## Compute FWHM in image #################################################
##################################################################################################################
majorsigma = np.array([])
sigx=np.array([])
sigy=np.array([])
for i in np.arange(0,len(sigmax)):
    sigx = np.append(sigx,sigmax[i])
for i in np.arange(0,len(sigmay)):
    sigy = np.append(sigx,sigmay[i])
    
# Find the take the larger of the two sigma dimensions:
for i,j in zip(sigx,sigy):
    biggestsigma = np.max([i,j])
    majorsigma = np.append(majorsigma,biggestsigma)

# Compute FWHM as 2.355* the major axis sigma:
FWHM = 2.355*majorsigma*pixscale
FWHMmean,FWHMstd = np.mean(FWHM),np.std(FWHM)

##################################################################################################################
######################################### Compute signal to noise ################################################
##################################################################################################################
# S/N is computed as the ratio of the volume under the narrow diffraction limited Gaussian to the std dev in that
# volume.  Gaussian volume is found as the noramlization constant for 2-d gaussian normalization: 2*pi*sigmax*sigmay
# times the amplitude of the gaussian.

vols = 2.*np.pi*sigmax*sigmay*(amps-bkgd)
volc = 2.*np.pi*sigmax*sigmay*(ampc-bkgd)

starsn = np.mean(vols)/np.std(vols)
compsn = np.mean(volc)/np.std(volc)


##################################################################################################################
############################################## Write to file #####################################################
##################################################################################################################

# File for import into positions analyzer script:
strg = str(imhdr['KOAID']) + ' , '
strg += str(sep_mean)
strg += ' , '
strg += str(sep_stdev)
strg += ' , '
strg += str(pa_mean)
strg += ' , '
strg += str(pa_stdev) + ' , '
strg += str(starsn) + ' , '
strg += str(compsn) + ' , '
strg += str(FWHMmean) + ' , '
strg += str(FWHMstd)

directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/epoch_positions_olpefit_pasep'

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

#  Write out RA/Dec positions to file:
directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/epoch_positions_olpefit_radec'

strg = str(RA)
strg += ' , '
strg += str(RAstd)
strg += ' , '
strg += str(Dec)
strg += ' , '
strg += str(Decstd)

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/epoch_positions_olpefit_log'

#Log file of all results:
b= imhdr['SAMPMODE'],imhdr['MULTISAM'],imhdr['COADDS'],imhdr['ITIME']
z = open(directory, 'a')
string = str(imhdr['KOAID']) + "\n"
string += ' comp pixel location: '
string += str(np.median(xcc))
string += ' , '
string += str(np.median(ycc)) + "\n"
string += ' star pixel location: '
string += str(np.median(xcs))
string += ' , '
string += str(np.median(ycs)) + "\n"
string += ' sep/PA: '
string += str(sep_mean)
string += ' , '
string += str(pa_mean) + "\n"
string += ' sep/PA std devs: '
string += str(sep_stdev)
string += ' , '
string += str(pa_stdev) + "\n"
string += ' RA/Dec: '+ str(RA)+' , ' + str(Dec) + "\n"
string += ' RA/Dec std devs: ' + str(RAstd) +' , ' + str(Decstd) + "\n"
string += ' NSamples: ' + str((length-additional_burnin)*ncor) + "\n"
string += ' Star S/N: ' + str(np.mean(vols)/np.std(vols)) + "\n"
string += ' Comp S/N: ' + str(np.mean(volc)/np.std(volc)) + "\n"
string += ' Samp mode,Multisam,Coadds,Itime: '
string += str(b) + "\n"
string += ' Vertical angle mode: '
string += a + "\n"
z.write(string + "\n")
z.close()

##################################################################################################################
############################################# Make a corner plot #################################################
##################################################################################################################
print 'Creating a corner plot...'
# Convert to numpy arrays and flatten:
xcs,ycs,xcc,ycc = np.array(xcs),np.array(ycs),np.array(xcc),np.array(ycc)
dx,dy = np.array(dx),np.array(dy)
amps,ampc,ampratio,bkgd = np.array(amps),np.array(ampc),np.array(ampratio),np.array(bkgd)
sigmax,sigmay,sigmax2,sigmay2,theta,theta2 = np.array(sigmax),np.array(sigmay),np.array(sigmax2),np.array(sigmay2),\
    np.array(theta),np.array(theta2)
sep,pa = np.array(sep),np.array(pa)
xcsf=xcs.flatten()
ycsf=ycs.flatten()
xccf=xcc.flatten()
yccf=ycc.flatten()
dxf=dx.flatten()
dyf=dy.flatten()
ampsf=amps.flatten()
ampcf=ampc.flatten()
ampratiof = ampratio.flatten()
bkgdf = bkgd.flatten()
sigmaxf = sigmax.flatten()
sigmayf = sigmay.flatten()
sigmay2f = sigmay2.flatten()
sigmax2f = sigmax2.flatten()
thetaf = theta.flatten()
theta2f = theta2.flatten()
sepf = sep.flatten()
paf = pa.flatten()

cornerplot = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_mpi'

# Make corner plot using "corner":
import corner
minshape = min(xcsf.shape,ycsf.shape,xccf.shape,yccf.shape,ampsf.shape,ampcf.shape,ampratiof.shape,bkgd.shape)
ndim, nsamples = 18, minshape
data = np.vstack([xcsf,ycsf,xccf,yccf,dxf,dyf,ampsf,ampcf,ampratiof,bkgdf,sigmaxf,sigmayf,sigmax2f,sigmay2f,thetaf,theta2f,sepf,paf])
data=data.transpose()
# Plot it.
plt.rcParams['figure.figsize'] = (10.0, 6.0)
figure = corner.corner(data, labels=["xcs", 'ycs', "xcc","ycc",'dx','dy','amps','ampc','ampratio','bkgd','sigmax','sigmay',\
                                    'sigmax2','sigmay2','theta','theta2','sep','pa'],
                       show_titles=True, plot_contours=True, title_kwargs={"fontsize": 12})
figure.savefig(cornerplot+'/'+filename.split('.')[2]+'_cornerplot', dpi=100)
print 'Done.'

