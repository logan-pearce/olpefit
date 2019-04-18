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

# example: python apf_step3.py -s 6 -a 8000 "1RXS J1609" ../1RXSJ1609/2009/N2.20090531.29966.LDIF.fits <- open the
                 step 2 output from this image which had been run using 6 processes, and disregard the
                 the first 8000 steps in the chain.
           python apf_step3_3body.py ../IC348-25/2013/N2.20130806.50421.LDIF.fits "Cl* IC 348 LRL 25" -s 6 -a 7000
                  <- perform step 3 corrections on the system IC 348-25 with a burn in of 7000 and 6 walkers


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
parser.add_argument("-a","--additional_burnin", help="Additional burn in to applly to chains",type=int)

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
print directory
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
print 'Parameter array shape:',c.shape[1]
index = ncor
xca,yca,xcb,ycb,xcc,ycc = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
dx,dy=np.zeros([length,ncor]),np.zeros([length,ncor])
ampa,ampb,ampc,ampratio,bkgd = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
sigmax,sigmay,sigmax2,sigmay2,theta,theta2,chisquare = np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),\
    np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor]),np.zeros([length,ncor])
# Place each walker's parameter into a collated parameter array - the columns of the array are each walker
for i in range(index):
    a = np.genfromtxt(input_directory+'/'+str(i)+'_finalarray_mpi.csv',delimiter=',')
    a = np.transpose(a)
    xca[:,i],yca[:,i],xcb[:,i],ycb[:,i],xcc[:,i],ycc[:,i] = a[0],a[1],a[2],a[3],a[4],a[5]
    dx[:,i],dy[:,i] = a[6],a[7]
    ampa[:,i],ampb[:,i],ampc[:,i],ampratio[:,i],bkgd[:,i] = a[8],a[9],a[10],a[11],a[12]
    sigmax[:,i],sigmay[:,i],sigmax2[:,i],sigmay2[:,i],theta[:,i],theta2[:,i],chisquare[:,i] = a[13],a[14],a[15],a[16],a[17],a[18],a[19]

#Determine the number of walkers used in the fit:
NWalkers = ncor

# Give the parameters some additional burn in:
xca = xca[additional_burnin:length,:] 
yca = yca[additional_burnin:length,:]
xcb = xcb[additional_burnin:length,:] 
ycb = ycb[additional_burnin:length,:]
xcc = xcc[additional_burnin:length,:] 
ycc = ycc[additional_burnin:length,:]
dx = dx[additional_burnin:length,:]
dy = dy[additional_burnin:length,:]
ampa = ampa[additional_burnin:length,:]
ampb = ampb[additional_burnin:length,:]
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
print "Number of jumps after additional burn in:",xca.shape[0]
print "Number of total samples after burn-in: ",xca.shape[0]*ncor

# Add one pixel to each value because python indexes starting at zero, and fits files start at 1 (for the distortion lookup table):
xca = xca+1 
yca = yca+1
xcb = xcb+1 
ycb = ycb+1
xcc = xcc+1
ycc = ycc+1

#######################################################################################################
####################################### Correct plate distortion ######################################
#######################################################################################################
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
xca_int = np.int_(xca)
yca_int = np.int_(yca)
xcb_int = np.int_(xcb)
ycb_int = np.int_(ycb)
xcc_int = np.int_(xcc)
ycc_int = np.int_(ycc)

# If the image is sized differently than 1024x1024, adjust the lookup indices to grab the correct distortion correction
#value:
xdiff=int(np.ceil(0.5*(1024-image.shape[1])))
ydiff=int(np.ceil(0.5*(1024-image.shape[0])))

xca_int = xca_int+xdiff
xcb_int = xcb_int+xdiff
xcc_int = xcc_int+xdiff
yca_int = yca_int+ydiff
ycb_int = ycb_int+ydiff
ycc_int = ycc_int+ydiff

# Add the distortion solution correction to each datapoint in the position arrays:
xca_dedistort = xca + x_dist[yca_int,xca_int]
yca_dedistort = yca + y_dist[yca_int,xca_int]
xcb_dedistort = xcb + x_dist[ycb_int,xcb_int]
ycb_dedistort = ycb + y_dist[ycb_int,xcb_int]
xcc_dedistort = xcc + x_dist[ycc_int,xcc_int]
ycc_dedistort = ycc + y_dist[ycc_int,xcc_int]

deltayb = ycb_dedistort-yca_dedistort
deltaxb = xcb_dedistort-xca_dedistort
deltayc = ycc_dedistort-yca_dedistort
deltaxc = xcc_dedistort-xca_dedistort

# Compute relative RA/Dec with star at 0,0 in pixel space:
# Convert to RA/Dec in milliarcseconds:
RAb = deltaxb*pixscale #Neg because RA is defined increasing right to left
Decb = deltayb*pixscale
RAc = deltaxc*pixscale 
Decc = deltayc*pixscale

RAb,RAstdb = -np.mean(RAb),np.std(RAb)
Decb,Decstdb = np.mean(Decb),np.std(Decb)
RAc,RAstdc = -np.mean(RAc),np.std(RAc)
Decc,Decstdc = np.mean(Decc),np.std(Decc)

#######################################################################################################
############################### Compute Gelman-Rubin Stats for chains #################################
#######################################################################################################

parameters = [xca,yca,xcb,ycb,xcc,ycc,dx,dy,ampa,ampb,ampc,ampratio,bkgd,sigmax,sigmay,sigmax2,sigmay2,theta,theta2]

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
print 'GR for positions:',RC[0],RC[1],RC[2],RC[3],RC[4],RC[5]

#######################################################################################################
######################################## Compute separation ###########################################
#######################################################################################################
print 'Computing separation and position angle...'
rsquareb = (deltayb**2)+(deltaxb**2)
rsquarec = (deltayc**2)+(deltaxc**2)
rb = np.sqrt(rsquareb)
rc = np.sqrt(rsquarec)
# ^ separation in pixel space
sepb = rb*pixscale # Convert to mas
sepc = rc*pixscale 

#######################################################################################################
####################################### Compute position angle ########################################
#######################################################################################################

def compute_pa(deltax,deltay):
    '''Compute position angle in degrees from deltax/deltay coords, applying
      appropriate NIRC2 corrections'''
    pa = np.arctan2(-deltax,deltay)
    pa = np.degrees(pa) % 360
    
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
        vertical_angle_mode = str('no')
        pass
    else:
        #Calculate exposure total length:
        vertical_angle_mode = str('yes')
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
    return pa,a

pab,vertical_angle_mode = compute_pa(deltaxb,deltayb)
pac,vertical_angle_mode = compute_pa(deltaxc,deltayc)

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

def atmospheric_correction(xcs_dedistort, ycs_dedistort, xcc_dedistort, ycc_dedistort, sep, pa):
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
        pa_mas = (sep_corrected)*(np.tan(np.radians(pa)))
        deltapa = deltaR * np.sin(np.radians(180. - (pa - parantel)))
        pa_mas_corrected = pa_mas + deltapa
        # Convert back into angle:
        pa_angle = np.arctan(pa_mas_corrected/sep_corrected)
        pa_angle = np.degrees(pa_angle)
        pa_angle[np.where(pa_angle<0)] = pa_angle[np.where(pa_angle<0)] + 180.
        '''
        if np.mean(pa) >= 90.:
            print 'greater than 90'
            pa_angle = np.degrees(pa_angle)+180.
            print pa_angle
        else:
            print 'less than 90'
            pa_angle = np.degrees(pa_angle)
            print pa_angle'''
    else:  # The companion is "above" the star
        # Apply to separation:
        deltasep = deltaR * np.cos(np.radians(pa - parantel))
        sep_corrected = sep + deltasep
        # Apply to angular distance:
        pa_mas = (sep_corrected)*np.tan(np.radians(pa))
        deltapa = deltaR * np.sin(np.radians(pa - parantel))
        # Convert back into angle:
        pa_mas_corrected = pa_mas + deltapa
        if pop > 270. and np.mean(pa) > 270:
            pa_angle = np.degrees(np.arctan(pa_mas_corrected/sep_corrected))+360.
        elif pop > 270. and np.mean(pa) < 270:
            pa_angle = np.degrees(np.arctan(pa_mas_corrected/sep_corrected))+180.
        else:
            pa_angle = np.degrees(np.arctan(pa_mas_corrected/sep_corrected))
    return sep_corrected,pa_angle

sep_correctedb,pa_angleb = atmospheric_correction(xca_dedistort,yca_dedistort,xcb_dedistort,ycb_dedistort, sepb, pab)
sep_correctedc,pa_anglec = atmospheric_correction(xca_dedistort,yca_dedistort,xcc_dedistort,ycc_dedistort, sepc, pac)

####################################### Compute mean sep and pa value ############################################

print 'Computing median values...'
sep_meanb,sep_stdevb = np.median(sep_correctedb),np.std(sep_correctedb)
pa_meanb,pa_stdevb = np.median(pa_angleb),np.std(pa_angleb)
print "rb = ", sep_meanb, "pab = ", pa_meanb

sep_meanc,sep_stdevc = np.median(sep_correctedc),np.std(sep_correctedc)
pa_meanc,pa_stdevc = np.median(pa_anglec),np.std(pa_anglec)
print "rc = ", sep_meanc, "pac = ", pa_meanc

####################################### Compute delta RA and delta Dec ###########################################

# Compute relative RA/Dec with star at 0,0 in pixel space:
# Convert to RA/Dec in milliarcseconds:

RAb = sep_correctedb * np.sin(np.radians(pa_angleb))
Decb = sep_correctedb * np.cos(np.radians(pa_angleb))

RAb,RAstdb = np.mean(RAb),np.std(RAb)
Decb,Decstdb = np.mean(Decb),np.std(Decb)

RAc = sep_correctedc * np.sin(np.radians(pa_anglec))
Decc = sep_correctedc * np.cos(np.radians(pa_anglec))

RAc,RAstdc = np.mean(RAc),np.std(RAc)
Decc,Decstdc = np.mean(Decc),np.std(Decc)

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
ax,ay=(np.int_(np.mean(xca)) , np.int_(np.mean(yca)))
bx,by=(np.int_(np.mean(xcb)) , np.int_(np.mean(ycb)))
cx,cy=(np.int_(np.mean(xcc)) , np.int_(np.mean(ycc)))

radius=5
r_in,r_out = 11,14

def compute_snr(x,y):
    positions = (x-1,y-1)
    ap = CircularAperture(positions,r=radius)
    skyan = CircularAnnulus(positions,r_in=11,r_out=14)
    ap = CircularAperture(positions,r=radius)
    skyan = CircularAnnulus(positions,r_in=11,r_out=14)
    apsum = ap.do_photometry(image)[0]
    skysum = skyan.do_photometry(image)[0]
    averagesky = skysum/skyan.area()
    signal = (apsum - ap.area()*averagesky)[0]
    n = ap.area()
    box = image[y+12:y+12+15,x+12:x+12+15]
    noise = np.std(box)
    noise = noise*np.sqrt(n)
    snr = signal/noise
    return snr

snra = compute_snr(ax,ay)
snrb = compute_snr(bx,by)
snrc = compute_snr(cx,cy)

############################################## Write to file #####################################################


# File for import into positions analyzer script:
strg = str(imhdr['KOAID']) + ' , '
strg += str(sep_meanb)+' , '
strg += str(sep_stdevb)+' , '
strg += str(pa_meanb)+' , '
strg += str(pa_stdevb) + ' , '
strg += str(sep_meanc)+' , '
strg += str(sep_stdevc)+' , '
strg += str(pa_meanc)+' , '
strg += str(pa_stdevc) + ' , '
strg += str(snra) + ' , '
strg += str(snrb) + ' , '
strg += str(snrc) + ' , '
strg += str(FWHMmean) + ' , '
strg += str(FWHMstd) + ' , '
strg += str(np.mean(RC)) + ' , '
strg += str(np.std(RC))

directory = output_directory+'epoch_grand_pasep'

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

#  Write out RA/Dec positions to file:
directory = output_directory+'epoch_grand_radec'

strg = str(imhdr['KOAID']) + ' , '
strg += str(RAb)+' , '
strg += str(RAstdb)+' , '
strg += str(Decb)+' , '
strg += str(Decstdb)+' , '
strg += str(RAc)+' , '
strg += str(RAstdc)+' , '
strg += str(Decc)+' , '
strg += str(Decstdc)

f = open(directory, 'a')
f.write(strg + "\n")
f.close()

directory = output_directory+'epoch_grand_log'

#Log file of all results:
b= imhdr['SAMPMODE'],imhdr['MULTISAM'],imhdr['COADDS'],imhdr['ITIME']
z = open(directory, 'a')
string = str(imhdr['KOAID']) + '  '+ now + "\n"
string += ' a pixel location: '
string += '  x:'+str(np.mean(xca))+' , '+str(np.std(xca))+ "\n"
string += '  y:'+str(np.mean(yca))+' , '+str(np.std(yca))+ "\n"
string += ' b pixel location: '
string += '  x:'+str(np.mean(xcb))+' , '+str(np.std(xcb))+ "\n"
string += '  y:'+str(np.mean(ycb))+' , '+str(np.std(ycb))+ "\n"
string += ' c pixel location: '
string += '  x:'+str(np.mean(xcc))+' , '+str(np.std(xcc))+ "\n"
string += '  y:'+str(np.mean(ycc))+' , '+str(np.std(ycc))+ "\n"
string += ' GR stats:'+str(RC) + "\n"
string += ' separation b: '+str(sep_meanb)+' , '+str(sep_stdevb)+ "\n"
string += ' position angle b: '+str(pa_meanb)+' , '+str(pa_stdevb)+ "\n"
string += ' RA/Dec b: '+ str(RAb)+' , ' + str(Decb) + "\n"
string += ' RA/Dec std devs b: ' + str(RAstdb) +' , ' + str(Decstdb) + "\n"
string += ' separation c: '+str(sep_meanc)+' , '+str(sep_stdevc)+ "\n"
string += ' position angle c: '+str(pa_meanc)+' , '+str(pa_stdevc)+ "\n"
string += ' RA/Dec c: '+ str(RAc)+' , ' + str(Decc) + "\n"
string += ' RA/Dec std devs c: ' + str(RAstdc) +' , ' + str(Decstdc) + "\n"

string += ' Number of total jumps per walker: ' + str(length) + "\n"
string += ' Burn in setting: ' + str(additional_burnin) + "\n"
string += ' Number of jumps per walker after additional burn in: ' + str(xca.shape[0]) + "\n"
string += ' Number of total samples per parameter: ' + str(xca.shape[0]*ncor) + "\n"

string += ' FWHM: '+str(FWHMmean)+' , '+str(FWHMstd)+ "\n"
string += ' A S/N: ' + str(snra) + "\n"
string += ' B S/N: ' + str(snrb) + "\n"
string += ' C S/N: ' + str(snrc) + "\n"
string += ' Samp mode,Multisam,Coadds,Itime: '
string += str(b) + "\n"
string += ' Vertical angle mode: '
string += str(vertical_angle_mode) + "\n"
z.write(string + "\n")
z.close()


############################################## Plot the chains #################################################
print ('Plotting chains....')
c = np.genfromtxt(input_directory+'/0_finalarray_mpi.csv',delimiter=',') # This just measures the size of the
# outputted parameter arrays
labels = ['xca','yca','xcb','ycb','xcc','ycc','dx','dy','ampa','ampb','ampc','ampratio','bkgd','sigmax',\
          'sigmay','sigmax2','sigmay2','theta','theta2','chisquared']
length = c.shape[0]

parameters = np.zeros([length,ncor,c.shape[1]])
for i in range(ncor):
    a = np.genfromtxt(input_directory+'/'+str(i)+'_finalarray_mpi.csv',delimiter=',')
    for j in range(c.shape[1]):
        parameters[:,i,j] = a[:,j]

plt.figure(figsize = (8,15))
for i in range(0,c.shape[1]):
    plt.subplot(c.shape[1]/2 + 1, 2, i+1)
    for j in range(ncor):
        plt.plot(range(parameters.shape[0]),parameters[:,j,i])
    plt.title(labels[i])
plt.tight_layout()
plt.savefig(input_directory+args.image.split('/')[-1].split('.')[-3]+'_chains', dpi=100)

############################################# Make a corner plot #################################################

print 'Creating a corner plot...'
# Convert to numpy arrays and flatten:
xca,yca,xcb,ycb,xcc,ycc = np.array(xca),np.array(yca),np.array(xcb),np.array(ycb),np.array(xcc),np.array(ycc)
dx,dy = np.array(dx),np.array(dy)
ampa,ampb,ampc,ampratio,bkgd = np.array(ampa),np.array(ampb),np.array(ampc),np.array(ampratio),np.array(bkgd)
sigmax,sigmay,sigmax2,sigmay2,theta,theta2 = np.array(sigmax),np.array(sigmay),np.array(sigmax2),np.array(sigmay2),\
    np.array(theta),np.array(theta2)
sepb,pab = np.array(sepb),np.array(pab)
sepc,pac = np.array(sepc),np.array(pac)
xcaf=xca.flatten()
ycaf=yca.flatten()
xcbf=xcb.flatten()
ycbf=ycb.flatten()
xccf=xcc.flatten()
yccf=ycc.flatten()
dxf=dx.flatten()
dyf=dy.flatten()
ampaf=ampa.flatten()
ampbf=ampb.flatten()
ampcf=ampc.flatten()
ampratiof = ampratio.flatten()
bkgdf = bkgd.flatten()
sigmaxf = sigmax.flatten()
sigmayf = sigmay.flatten()
sigmay2f = sigmay2.flatten()
sigmax2f = sigmax2.flatten()
thetaf = theta.flatten()
theta2f = theta2.flatten()
sepbf = sepb.flatten()
pabf = pab.flatten()
sepcf = sepc.flatten()
pacf = pac.flatten()

# Make corner plot using "corner":
import corner
minshape = min(xcaf.shape,ycaf.shape,xcbf.shape,ycbf.shape,xccf.shape,yccf.shape,ampaf.shape,ampbf.shape,ampcf.shape,ampratiof.shape,bkgd.shape)
ndim, nsamples = 18, minshape
data = np.vstack([xcaf,ycaf,xcbf,ycbf,xccf,yccf,dxf,dyf,ampaf,ampbf,ampcf,ampratiof,bkgdf,sigmaxf,sigmayf,sigmax2f,sigmay2f,thetaf,theta2f,sepbf,pabf,sepcf,pacf])
data=data.transpose()
# Plot it.
plt.rcParams['figure.figsize'] = (10.0, 6.0)
figure = corner.corner(data, labels=["xca", 'yca', 'xcb', 'ycb', "xcc","ycc",'dx','dy','ampa','ampb','ampc','ampratio','bkgd','sigmax','sigmay',\
                                    'sigmax2','sigmay2','theta','theta2','sepb','pab','sepc','pac'],
                       show_titles=True, plot_contours=True, title_kwargs={"fontsize": 12})
figure.savefig(input_directory+args.image.split('/')[-1].split('.')[-3]+'_cornerplot', dpi=100)

print 'Donezo.'



