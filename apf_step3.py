'''
############################ LAPF: Logan's Analytical PSF Fitter ##############################
                                          Step 3
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
#   python packages astropy, numpy, matplotlib, photutils, photometry (my script located on github)
#
# Input:
#   Step 1: Image (fits file)
#   Step 2: Image (fits file), output from step 1 (text file)
#   Step 3: Output from step 2 (.csv files from each process)
#
# Output:
#   Step 3: 
#
# usage: apf_step3.py [-h] [-s SIZE] [-a ADDITIONAL_BURNIN]
                      directory

positional arguments:
  directory             the path to the directory containing LAPF step 2 results
  system                the name of the system to look up in Simbad

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Number of processes.
  -a ADDITIONAL_BURNIN, --additional_burnin ADDITIONAL_BURNIN
                        Additional burn in to applly to chains

# example: python apf_step3.py -s 6 -a 8000 ../1RXSJ1609/2009/N2.20090531.29966.LDIF.fits <- open the
                 step 2 output from this image which had been run using 6 processes, and disregard the
                 the first 8000 steps in the chain.


'''


import numpy as np
import argparse
from astropy.io import fits
from astropy.io.fits import getheader
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u
import photutils
from photutils import CircularAperture, CircularAnnulus
import warnings
import photometry

warnings.filterwarnings('ignore')

############################################################################

# Pull out arguments:
parser = argparse.ArgumentParser()
# Required positional arguments:
parser.add_argument("image", help="the path to the image under study", type=str)
parser.add_argument("system", help="name of the system for look up in Simbad", type=str)
# Optional positional arguments"
parser.add_argument("-s","--size", help="Number of processes",type=str)
parser.add_argument("-a","--additional_burnin", help="Additional burn in to applly to chains",type=str)

args = parser.parse_args()

if args.size:
    ncor = np.int_(args.size)
else:
    ncor = np.float(raw_input('Please input number of processes used in fit: '))

if args.additional_burnin:
    additional_burnin = args.additional_burnin
else:
    additional_burnin = 1

########################### Definitions: ####################################
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
        print 'Atmospheric correction not required, zenith greater than 70 deg'
        return 0.,0.


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

##############################################################################

# Open step 2 output:
# Get directory to store output:
d = args.image.split('/')
directory = ''
for i in range(len(d)-1):
    directory = directory + str(d[i]) + '/'
    
input_directory =  directory + d[-1].split('.')[-3] + '_apf_results/'
# Place the output in the epoch's folder:
output_directory = directory

# Determine which distortion solution to use:
epoch = int(d[-2])
if epoch >= 2015:
    prepost = "post"
else:
    prepost = "pre"

# get current date:
now = str(Time.now())
now=now.split(' ')[0] # give just the date without the time

print 'Burn in:',additional_burnin

################################# Open image: #################################

image = fits.open(args.image)[0].data
imhdr = getheader(args.image)

#################################### Import step 2 output: ######################

print 'Importing parameter arrays...'
c = np.genfromtxt(input_directory+'/0_finalarray_mpi.csv',delimiter=',') # This just measures the size of the
# outputted parameter arrays
c = np.transpose(c)
length = c.shape[1]
print c.shape
xcs,ycs,xcc,ycc = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
dx,dy=np.zeros([length,ncor]),np.zeros([length,ncor])
amps,ampc,ampratio,bkgd = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chisquare = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),\
    np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
# Place each walker's parameter into a collated parameter array - the columns of the array are each walker
for i in range(ncor):
    a = np.genfromtxt(input_directory+'/'+str(i)+'_finalarray_mpi.csv',delimiter=',')
    a = np.transpose(a)
    xcs[:,i],ycs[:,i],xcc[:,i],ycc[:,i] = a[0],a[1],a[2],a[3]
    dx[:,i],dy[:,i] = a[4],a[5]
    amps[:,i],ampc[:,i],ampratio[:,i],bkgd[:,i] = a[6],a[7],a[8],a[9]
    sigmax[:,i],sigmay[:,i],sigmax2[:,i],sigmay2[:,i],theta[:,i],theta2[:,i],chisquare[:,i] = a[10],a[11],a[12],a[13],a[14],a[15],a[16]

# Give the parameters some additional burn in:
xcs = xcs[additional_burnin:length,:] 
ycs = ycs[additional_burnin:length,:]
xcc = xcc[additional_burnin:length,:] 
ycc = ycc[additional_burnin:length,:]
dx = dx[additional_burnin:length,:]
dy = dy[additional_burnin:length,:]
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
print "Number of jumps after additional burn in:",xcs.shape[0]
print "Number of total samples after burn-in: ",xcs.shape[0]*ncor

# Add one pixel to each value because python indexes starting at zero, and fits files start at 1 (for the distortion lookup table):
xcs = xcs+1 
ycs = ycs+1
xcc = xcc+1
ycc = ycc+1

####################################### Correct plate distortion ######################################
print 'Correcting plate scale distortion...'
if prepost == 'pre' or prepost =='Pre':
    # Open the lookup tables of Yelda 2010:
    x_dist = fits.open('/Users/loganpearce/Dropbox/UTexas_research/nirc2_X_distortion.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('/Users/loganpearce/Dropbox/UTexas_research/nirc2_Y_distortion.fits')
    y_dist = y_dist[0].data
    pixscale = 9.952 #In mas
elif prepost == 'post' or prepost =='Post':
    # Open the lookup tables of Service 2016:
    x_dist = fits.open('/Users/loganpearce/Dropbox/UTexas_research/nirc2_distort_X_post20150413_v1.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('/Users/loganpearce/Dropbox/UTexas_research/nirc2_distort_Y_post20150413_v1.fits')
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

deltay = ycc_dedistort-ycs_dedistort
deltax = xcc_dedistort-xcs_dedistort

# Compute relative RA/Dec with star at 0,0 in pixel space:
# Convert to RA/Dec in milliarcseconds:
RA = deltax*pixscale #Neg because RA is defined increasing right to left
Dec = deltay*pixscale

RA,RAstd = -np.mean(RA),np.std(RA)
Dec,Decstd = np.mean(Dec),np.std(Dec)

############################### Compute Gelman-Rubin Stats for chains #################################

parameters = [xcs,ycs,xcc,ycc,dx,dy,amps,ampc,ampratio,bkgd,sigmax,sigmay,sigmax2,sigmay2,theta,theta2]

N,M = float(length-additional_burnin),float(ncor)
PSRF,RC = np.zeros(len(parameters)),np.zeros(len(parameters))
d = 16
for p,j in zip(parameters,range(len(parameters))):
    w,b = np.zeros(ncor),np.zeros(ncor)
    overall_mean = np.mean(p)
    for i in range(ncor):
        chain_mean = np.mean(p[:,i])
        w[i] = np.std(p[:,i])**2
        b[i] = (chain_mean - overall_mean)**2
    w = (1./M)*np.sum(w)
    b = (N/(M-1))*np.sum(b)
    pooled_variance = ((N-1)/N)*w + ((M+1)/(M*N))*b
    PSRF[j] = pooled_variance/w
    RC[j] = np.sqrt(((d+3)/(d+1))*PSRF[j])
print 'Mean and stdev Gelman-Rubin stat for parameter chains:',np.mean(RC),np.std(RC)

######################################## Compute separation ###########################################

print 'Computing separation and position angle...'
rsquare = (deltay**2)+(deltax**2)
r = np.sqrt(rsquare)
# ^ separation in pixel space
sep = r*pixscale # Convert to mas

####################################### Compute position angle ########################################

pa = np.arctan2(-deltax,deltay)
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

####################################### Atmospheric distortion correction ########################################

print 'Correcting atmospheric distortion...'
# Get zenith angle of star:
# Get Keck location:
keck = EarthLocation.of_site('Keck Observatory')
# Get RA/Dec of host star using Sesame:
gsc = SkyCoord.from_name(args.system)
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


####################################### Compute mean sep and pa value ############################################

print 'Computing median values...'
sep_mean,sep_stdev = np.median(sep_corrected),np.std(sep_corrected)
pa_mean,pa_stdev=np.median(pa_angle),np.std(pa_angle)

print "r = ", sep_mean, "pa = ", pa_mean

########################################## Compute FWHM in image #################################################

majorsigma = np.array([])
sigx=np.array([])
sigy=np.array([])
# Just take 100 data points to compute average FWHM.
for i in np.arange(len(sigmax)-100,len(sigmax)):
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

######################################### Compute signal to noise ########################################

import photometry

cx,cy=(np.int_(np.mean(xcc)) , np.int_(np.mean(ycc)))
sx,sy=(np.int_(np.mean(xcs)) , np.int_(np.mean(ycs)))
radius=5
r_in,r_out = 11,14

# Companion:
positions = (cx-1,cy-1)
ap = CircularAperture(positions,r=radius)
skyan = CircularAnnulus(positions,r_in=11,r_out=14)
ap = CircularAperture(positions,r=radius)
skyan = CircularAnnulus(positions,r_in=11,r_out=14)
apsum = ap.do_photometry(image)[0]
skysum = skyan.do_photometry(image)[0]
averagesky = skysum/skyan.area()
signal = (apsum - ap.area()*averagesky)[0]
n = ap.area()
box = image[cy+12:cy+12+15,cx+12:cx+12+15]
noise = np.std(box)
noise = noise*np.sqrt(n)
compsnr = signal/noise

# Star:
positions = (sx-1,sy-1)
ap = CircularAperture(positions,r=radius)
skyan = CircularAnnulus(positions,r_in=11,r_out=14)
ap = CircularAperture(positions,r=radius)
skyan = CircularAnnulus(positions,r_in=11,r_out=14)
apsum = ap.do_photometry(image)[0]
skysum = skyan.do_photometry(image)[0]
averagesky = skysum/skyan.area()
signal = (apsum - ap.area()*averagesky)[0]
n = ap.area()
box = image[sy+12:sy+12+15,sx+12:sx+12+15]
noise = np.std(box)
noise = noise*np.sqrt(n)
starsnr = signal/noise

############################################## Write to file #####################################################


# File for import into positions analyzer script:
strg = str(imhdr['KOAID']) + ' , '
strg += str(sep_mean)+' , '
strg += str(sep_stdev)+' , '
strg += str(pa_mean)+' , '
strg += str(pa_stdev) + ' , '
strg += str(starsnr) + ' , '
strg += str(compsnr) + ' , '
strg += str(FWHMmean) + ' , '
strg += str(FWHMstd) + ' , '
strg += str(np.mean(RC)) + ' , '
strg += str(np.std(RC))

directory = outdir+'epoch_grand_pasep'

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

#  Write out RA/Dec positions to file:
directory = outdir+'epoch_grand_radec'

strg = str(imhdr['KOAID']) + ' , '
strg += str(RA)+' , '
strg += str(RAstd)+' , '
strg += str(Dec)+' , '
strg += str(Decstd)

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

directory = outdir+'epoch_grand_log'

#Log file of all results:
b= imhdr['SAMPMODE'],imhdr['MULTISAM'],imhdr['COADDS'],imhdr['ITIME']
z = open(directory, 'a')
string = str(imhdr['KOAID']) + '  '+ now + "\n"
string += ' comp pixel location: '
string += '  x:'+str(np.mean(xcc))+' , '+str(np.std(xcc))+ "\n"
string += '  y:'+str(np.mean(ycc))+' , '+str(np.std(ycc))+ "\n"
string += ' star pixel location: '
string += '  x:'+str(np.mean(xcs))+' , '+str(np.std(xcs))+ "\n"
string += '  y:'+str(np.mean(ycs))+' , '+str(np.std(ycs))+ "\n"
string += ' GR stats:'+str(Rc) + "\n"
string += ' separation: '+str(sep_mean)+' , '+str(sep_stdev)+ "\n"
string += ' position angle: '+str(pa_mean)+' , '+str(pa_stdev)+ "\n"
string += ' RA/Dec: '+ str(RA)+' , ' + str(Dec) + "\n"
string += ' RA/Dec std devs: ' + str(RAstd) +' , ' + str(Decstd) + "\n"

string += ' Number of total jumps per walker: ' + str(length) + "\n"
string += ' Burn in setting: ' + str(additional_burnin) + "\n"
string += ' Number of jumps per walker after additional burn in: ' + str(xcs.shape[0]) + "\n"
string += ' Number of total samples per parameter: ' + str(xcs.shape[0]*ncor) + "\n"

string += ' FWHM: '+str(FWHMmean)+' , '+str(FWHMstd)+ "\n"
string += ' Star S/N: ' + str(starsnr) + "\n"
string += ' Comp S/N: ' + str(compsnr) + "\n"
string += ' Samp mode,Multisam,Coadds,Itime: '
string += str(b) + "\n"
string += ' Vertical angle mode: '
string += a + "\n"
z.write(string + "\n")
z.close()

############################################# Make a corner plot #################################################

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
figure.savefig(filepath+'/'+filename.split('.')[-3]+'_cornerplot', dpi=100)

print 'Donezo.'



