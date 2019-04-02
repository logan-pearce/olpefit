'''
##################### Common Proper Motion Plots ##########################
                     written by Logan Pearce, 2019
#######################################################################
    Test for common proper motion of a candidate companion (cc) in images by plotting
the motion the cc would have been observed on if it were a background star (the host star
is treated as not moving), compared to the postitions our relative astrometry observed it at.
If our cc does not appear to move consistent with the background star hypothesis, this supports 
the hypothesis that is a graviationally bound companion because it is exhibiting common 
proper motion with the host star.

# Requires:
#   python packages astropy, numpy, maplotplotlib
#   optional: mpl stylefile
#
# Input:
#   Directly input observations, and host star proper motion from Gaia
#
# Output:
#      common proper motion plot
#
# usage: make_squiggle_plot.py

'''


from astropy import units as u 
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
deg_to_mas = 3600000.
mas_to_deg = 1./3600000.

plt.style.use('supermongo')

############################### Observations ################################
# Object from Gaia:
RA, RAerr, Dec, Decerr = 245.47769209925707, 0.04335031149660691*mas_to_deg, \
    -20.719339784108172, 0.022706948867453742*mas_to_deg #Deg
    
pmRA,pmRAerr,pmDec,pmDecerr = -19.378835001368177, 0.10312946207539458, \
    -31.247630493393903, 0.07009716508634732 #mas/yr
    
parallax, parallaxerr = 9.187965139256493,0.04301793042607285 #mas

# Observation times for predicting location if cc were a background star:
obsdate = np.array([2008.46,2009.41,2010.40,2011.42,2014.58,2016.46,2017.49,2018.495])
labels = ['2008','2009','2010','2011','2014','2016','2017','2018']

# Relative astronomical observations (from astrom pipeline output):
epochs = ['2008_06_17','2010_04_26','2011_06_04','2016_06_16','2017_06_27','2018_07_01']
obs_RAs, obs_Decs = np.zeros(len(epochs)),np.zeros(len(epochs))
obs_RAs_err, obs_Decs_err = np.zeros(len(epochs)),np.zeros(len(epochs))
for i in range(len(epochs)):
    pos = np.loadtxt(open('GSC6214/data/'+epochs[i]+'/epoch_grand_radec',"rb")\
                 ,delimiter=",",usecols=(1,2,3,4))
    obs_RAs[i],obs_RAs_err[i] = np.mean(pos[:,0]),np.mean(pos[:,1])
    obs_Decs[i],obs_Decs_err[i] = np.mean(pos[:,2]),np.mean(pos[:,3])


# Specify a reference time:
time = Time(obsdate[0],format='decimalyear',scale='utc')

############################### Plot settings ###############################

output_name = 'GSC6214_cpm.pdf'
form = 'pdf'
time_interval = [12,12]
plt_xlim = [100,380]
plt_ylim = [-2250,-1800]
figsize =  (6,7.5)
n_times =  800
marker = ['^','o']
markersize = [130,100]
fontsize = 15
tick_labelsize = 12
labelsize = 13

############################### Definitions #################################
def ecliptic_to_equatorial(lon, lat):
    ''' Convert array from ecliptic to equatorial coordinates using astropy's SkyCoord object
        Inputs:
            lon, lat [deg] (array): ecliptic longitude (lambda) and ecliptic latitude (beta)
        Returns:
            newRA, newDec [deg] (array): array points in equatorial RA/Dec coordinates
    '''
    # Compute ecliptic motion to equatorial motion:
    newRA, newDec = np.zeros(len(lon)),np.zeros(len(lon))
    for i in range(len(lon)):
        obj2 = SkyCoord(lon = lon[i],\
                    lat = lat[i], \
                    frame='geocentrictrueecliptic', unit='deg') 
        obj2 = obj2.transform_to('icrs')
        newRA[i] = obj2.ra.deg
        newDec[i] = obj2.dec.deg
    return newRA,newDec
    

def squiggle_plot(RA, RAerr, Dec, Decerr, pmRA, pmRAerr, pmDec, pmDecerr, parallax, parallaxerr, ref_date, \
                  obsdate, obs_RAs, obs_RAs_err, obs_Decs, obs_Decs_err, labels, \
                  ref_RA_offset = 0,
                  ref_Dec_offset = 0,
                  time_interval = [12,12],
                  n_times = 5000,
                  plt_xlim=[-200,200],
                  plt_ylim=[-200,200],
                  marker = ['^','o'],
                  marker_size = [100,100],
                  figsize = (8,8)
                 ):
    ''' Test for common proper motion of a candidate companion by plotting the track the cc would have
        been observed on if it were a background object and not graviationally bound.
        Inputs:
            RA/Dec + errors [deg] (flt): RA/Dec of host star
            pmRA, pmDec + errors [mas/yr] (flt): - proper motion of host star.  Use the negative of reported
                values because we treat the star as unmoving and will observe the apparent motion of the cc
            parallax + erro [mas] (flt): parallax
            ref_date [decimal year] (astropy Time object): reference date
            obsdate [decimal year] (array): array of dates of observations
            obs_RAs, obs_Decs + errors [mas] (array): array of observed RA/Dec offsets of companion to host star
            labels (str array): strings to label plot points 
            ref_RA_offset, ref_Dec_offset [mas] (flt): 'zero point' reference RA/Dec offset for for companion
                to host
            time_interval [yrs] (array):  Number of years [below,above] reference date to compute plot
            n_times (int): number of time points to compute values for plot
            plt_xlim, plt_ylim [mas] (array): axis limits [min,max]
            marker (array): markers to use for prediction points [0] and observed points [1]
            marker_size (array): size of markers for predicition points [0] and observed points [1]
        Returns:
            fig (matplotlib figure): plot of proper motion track
            pred_dRA_total, pred_dDec_total (flt): predicted RA/Dec offsets if cc were a background object
    '''
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    ############### Compute track: ###################
    # Define a time span around reference date:
    delta_time = np.linspace(-time_interval[0], time_interval[1], n_times)*u.yr
    times = ref_date + delta_time
    
    # Compute change in RA/Dec during time interval due to proper motion only:
    dRA, dDec = (pmRA)*(delta_time.value), (pmDec)*(delta_time.value)
    
    # Compute motion in the ecliptic coords due to parallactic motion:
    # Make a sky coord object in RA/Dec:
    obj = SkyCoord(ra = RA, dec = Dec, frame='icrs', unit='deg', obstime = time) 
    # Convert to ecliptic lon/lat:
    gteframe = GeocentricTrueEcliptic()
    obj_ecl = obj.transform_to(gteframe)
    
    # Angle array during a year:
    theta = (delta_time.value%1)*2*np.pi
    #Parallel to ecliptic:
    x = parallax*np.sin(theta)  
    #Perp to ecliptic
    y = parallax*np.sin(obj_ecl.lat.rad)*np.cos(theta)  
    
    # Compute ecliptic motion to equatorial motion:
    print 'Plotting... this part may take a minute.'
    new_RA, new_Dec = ecliptic_to_equatorial(obj_ecl.lon.deg+x*mas_to_deg, \
                                           obj_ecl.lat.deg+y*mas_to_deg)
    # Compute change in RA/Dec for each time point in mas:
    delta_RA, delta_Dec = (new_RA-RA)*deg_to_mas,(new_Dec-Dec)*deg_to_mas
    
    #Put it together:
    dRA_total = delta_RA + dRA + ref_RA_offset
    dDec_total = delta_Dec + dDec + ref_Dec_offset
    
    ############# Compute prediction: #############
    ### Where the object would have been observed were it a background object
    
    # Compute how far into each year the observation occured:
    pred_time_delta = (obsdate - np.floor(obsdate))
    pred_theta = (pred_time_delta)*2*np.pi
    
    # Compute ecliptic motion:
    pred_x = parallax*np.sin(pred_theta)  #Parallel to ecliptic
    pred_y = parallax*np.sin(obj_ecl.lat.rad)*np.cos(pred_theta)  #Perp to ecliptic
    
    # Convert to RA/Dec:
    pred_new_RA, pred_new_Dec = ecliptic_to_equatorial(obj_ecl.lon.deg+pred_x*mas_to_deg, \
                                           obj_ecl.lat.deg+pred_y*mas_to_deg)
    pred_delta_RA, pred_delta_Dec = (pred_new_RA-RA)*deg_to_mas,(pred_new_Dec-Dec)*deg_to_mas
    
    # Compute location due to proper motion:
    pred_dRA, pred_dDec = (pmRA)*(obsdate-ref_date.value), (pmDec)*(obsdate-ref_date.value)

    # Put it together:
    pred_dRA_total = -pred_delta_RA + pred_dRA + ref_RA_offset
    pred_dDec_total = -pred_delta_Dec + pred_dDec + ref_Dec_offset

    #################### Draw plot: #################
    plt.rcParams['ytick.labelsize'] = tick_labelsize
    plt.rcParams['xtick.labelsize'] = tick_labelsize
    fig = plt.figure(figsize = figsize)
    plt.plot(dRA_total,dDec_total, lw=3, color='lightgrey', alpha = 0.5, zorder = 0)
    plt.plot(dRA_total,dDec_total, zorder = 1)
    for i in range(len(pred_dRA)):
        plt.scatter(pred_dRA_total[i], pred_dDec_total[i], marker = marker[0], s=marker_size[0], zorder=2, 
                    edgecolors='black')
        plt.annotate(
            labels[i],
            xy=(pred_dRA_total[i], pred_dDec_total[i]), xytext=(-10, 10),
            textcoords='offset points', ha='right', va='bottom', fontsize=labelsize)
    for i in range(len(obs_RAs)):
        plt.scatter(obs_RAs[i], obs_Decs[i], edgecolors="black", marker = marker[1], s=marker_size[1], zorder = 10)
        plt.errorbar(obs_RAs[i], obs_Decs[i], xerr= obs_RAs_err[i], yerr=obs_Decs_err[i], ls='none',
                 elinewidth=1,capsize=0, ecolor='black',zorder=10)
    plt.ylim(plt_ylim[0],plt_ylim[1])
    plt.xlim(plt_xlim[0],plt_xlim[1])
    plt.xlabel(r'$\Delta$ RA [mas]', fontsize = fontsize)
    plt.ylabel(r'$\Delta$ Dec [mas]', fontsize = fontsize)
    plt.grid(ls=':')
    plt.tight_layout()
    
    return fig, pred_dRA_total, pred_dDec_total

ax = squiggle_plot(RA, RAerr, Dec, Decerr, -pmRA, pmRAerr, -pmDec, pmDecerr, parallax, parallaxerr, time, \
                  obsdate, obs_RAs, obs_RAs_err, obs_Decs, obs_Decs_err, labels, \
                   ref_RA_offset = obs_RAs[0],
                   ref_Dec_offset = obs_Decs[0],
                   time_interval = time_interval,
                   n_times = n_times, plt_xlim=plt_xlim, 
                   plt_ylim=plt_ylim, marker = marker, marker_size = markersize,
                  figsize =figsize)
ax = ax[0]

######### write it out ###############
ax.savefig(output_name, format=form)
print 'Done'
