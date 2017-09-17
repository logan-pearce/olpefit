import numpy as np
import argparse
from astropy.io import fits
from astropy.io.fits import getheader
import matplotlib.pyplot as plt

# Get the file name from the entered argument
parser = argparse.ArgumentParser()
parser.add_argument("image_filename",type=str)
args = parser.parse_args()
filename=args.image_filename

imhdr = getheader(args.image_filename)

epoch = filename.split('/')[1]
epoch = epoch.split('_')[0]

if epoch >= 2015:
    prepost = "post"
else:
    prepost = "pre"
#prepost = raw_input("Was this observation pre or post April 13, 2015? Type 'pre' or 'post' ")

directory = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_tacc'

additional_burnin = 100

############################################### Calculate PA/Sep ##########################################
open_xcs = open(directory + '/xcsarray.csv', 'r')
c = open_xcs.read()
# read in the csv file, split the rows (the filter removes any empty lines):
d = filter(None,c.split('\n'))
# for each row ("i in range(len(d))"), split the row by commas, make each object a float, and place
# them into a numpy array.  Now each walker is callable as arrays within this array:
xcs = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
#print len(xcs[0])

open_ycs = open(directory + '/ycsarray.csv', 'r')
c = open_ycs.read()
d = filter(None,c.split('\n'))
ycs = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
#print len(ycs[0])

open_xcc = open(directory + '/xccarray.csv', 'r')
c = open_xcc.read()
d = filter(None,c.split('\n'))
xcc = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
#print len(xcc[0])

open_ycc = open(directory + '/yccarray.csv', 'r')
c = open_ycc.read()
d = filter(None,c.split('\n'))
ycc = [np.array([float(string) for string in d[i].split(',')]) for i in range(len(d))]
#print len(ycc[0])

e = open(directory + '/ampsarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
amps = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(amps[0])

e = open(directory + '/ampcarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
ampc = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(ampc[0])

e = open(directory + '/ampratioarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
ampratio = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(ampratio[0])

e = open(directory + '/bkgdarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
bkgd = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(bkgd[0])

e = open(directory + '/sigmaxarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
sigmax = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(sigmax[0]) 

e = open(directory + '/sigmayarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
sigmay = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(sigmay[0])

e = open(directory + '/sigmax2array.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
sigmax2 = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(sigmax2[0])

e = open(directory + '/sigmay2array.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
sigmay2 = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(sigmay2[0])

e = open(directory + '/thetaarray.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
theta = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(theta[0])

e = open(directory + '/theta2array.csv', 'r')
f = e.read()
g = filter(None,f.split('\n'))
theta2 = [np.array([float(string) for string in g[i].split(',')]) for i in range(len(g))]
#print len(theta2[0])

# Truncate all arrays to the size of the smallest one:
minxy = min(len(xcs[0]),len(ycs[0]),len(xcc[0]),len(ycc[0]),len(amps[0]),len(ampc[0]),len(ampratio[0]),len(bkgd[0]),\
           len(sigmax[0]),len(sigmay[0]),len(sigmax2[0]),len(sigmay2[0]),len(theta[0]),len(theta2[0]))

#Determine the number of walkers used in the fit:
NWalkers = len(xcs)

# Truncate the results to the size of the smallest parameter arrays so that array math can be performed:
xcs = [xcs[i][additional_burnin:minxy] for i in range(NWalkers)] 
ycs = [ycs[i][additional_burnin:minxy] for i in range(NWalkers)]
xcc = [xcc[i][additional_burnin:minxy] for i in range(NWalkers)] 
ycc = [ycc[i][additional_burnin:minxy] for i in range(NWalkers)]
amps = [amps[i][additional_burnin:minxy] for i in range(NWalkers)]
ampc = [ampc[i][additional_burnin:minxy] for i in range(NWalkers)]
ampratio = [ampratio[i][additional_burnin:minxy] for i in range(NWalkers)]
bkgd = [bkgd[i][additional_burnin:minxy] for i in range(NWalkers)]
sigmax = [sigmax[i][additional_burnin:minxy] for i in range(NWalkers)]
sigmay = [sigmay[i][additional_burnin:minxy] for i in range(NWalkers)]
sigmax2 = [sigmax2[i][additional_burnin:minxy] for i in range(NWalkers)]
sigmay2 = [sigmay2[i][additional_burnin:minxy] for i in range(NWalkers)]
theta = [theta[i][additional_burnin:minxy] for i in range(NWalkers)]
theta2 = [theta2[i][additional_burnin:minxy] for i in range(NWalkers)]


# Add one pixel to each value because python indexes starting at zero, and fits files start at 1 (for the distortion lookup table):
xcs = [xcs[i]+1 for i in range(NWalkers)] 
ycs = [ycs[i]+1 for i in range(NWalkers)]
xcc = [xcc[i]+1 for i in range(NWalkers)]
ycc = [ycc[i]+1 for i in range(NWalkers)]

####### Correct plate distortion:
if prepost == 'pre' or prepost =='Pre':
    # Open the lookup tables of Yelda 2010:
    x_dist = fits.open('../nirc2_X_distortion.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('../nirc2_Y_distortion.fits')
    y_dist = y_dist[0].data
elif prepost == 'post' or prepost =='Post':
    # Open the lookup tables of Service 2010:
    x_dist = fits.open('../nirc2_distort_X_post20150413_v1.fits')
    x_dist = x_dist[0].data
    y_dist = fits.open('../nirc2_distort_X_post20150413_v1.fits')
    y_dist = y_dist[0].data
        
# Convert pixel locations to integers to feed into lookup table:
xcc_int = [np.int_(xcc[i]) for i in range (NWalkers)]
ycc_int = [np.int_(ycc[i]) for i in range (NWalkers)]
xcs_int = [np.int_(xcs[i]) for i in range (NWalkers)]
ycs_int = [np.int_(ycs[i]) for i in range (NWalkers)]
# Add the distortion solution correction to each datapoint in the position arrays:
xcc_dedistort = [xcc[i] + x_dist[ycc_int[i],xcc_int[i]] for i in range(NWalkers)]
ycc_dedistort = [ycc[i] + y_dist[ycc_int[i],xcc_int[i]] for i in range(NWalkers)]
xcs_dedistort = [xcs[i] + x_dist[ycs_int[i],xcs_int[i]] for i in range(NWalkers)]
ycs_dedistort = [ycs[i] + y_dist[ycs_int[i],xcs_int[i]] for i in range(NWalkers)]

dy = [ycc_dedistort[i]-ycs_dedistort[i] for i in range(NWalkers)]
dx = [xcc_dedistort[i]-xcs_dedistort[i] for i in range(NWalkers)]

# Compute relative RA/Dec with star at 0,0 in pixel space:
pixscale = imhdr['PIXSCALE']
##### Convert to RA/Dec in milliarcseconds:
RA = [dx[i]*pixscale*1000 for i in range(NWalkers)] #Neg because RA is defined increasing right to left
Dec = [dy[i]*pixscale*1000 for i in range(NWalkers)]

RA,RAstd = -np.mean(RA),np.std(RA)
Dec,Decstd = np.mean(Dec),np.std(Dec)

######## Compute separation:
rsquare = [(dy[i]**2)+(dx[i]**2) for i in range(NWalkers)]
r = [np.sqrt(rsquare[i]) for i in range (NWalkers)]
# ^ separation in pixel space

pixscale = imhdr['PIXSCALE']
sep = [r[i]*pixscale*1000 for i in range(NWalkers)] #converted to milliarcseconds

######## Compute position angle:
# Compute position angle:
pa = [np.arctan2(-dx[i],dy[i]) for i in range(NWalkers)]
pa = [np.degrees(pa[i]) for i in range(NWalkers)]

# Rotation angle correction:
p = imhdr['PARANG']
r = imhdr['ROTPPOSN']
i = imhdr['INSTANGL']
e = imhdr['EL']

if prepost == 'pre' or prepost =='Pre':
    rotation_angle = p+r-e-i-0.252  # Yelda 2010 soln
elif prepost == 'post' or prepost =='Post':
    rotation_angle = p+r-e-i-0.262 # Service 2016 soln

pa = [pa[i]+rotation_angle for i in range(NWalkers)]


######## Vertical angle mode correction:
# Vertical angle mode correction:
mode = imhdr['ROTMODE']

if mode != 'vertical angle':
    #print 'vertical angle rotation compensation not required'
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
    pa = [pa[i]+rotcorr for i in range(NWalkers)]

# If the computed position angle is negative, add 360 deg:
if np.mean(pa) < 0:
    pa = [pa[i]+360. for i in range(NWalkers)]
else:
    pass

######## Compute mean sep and pa value:
sep,sep_stdev = np.mean(sep),np.std(sep)
pa,pa_stdev=np.mean(pa),np.std(pa)

print "r = ", sep, "pa = ", pa

### Write out PA/Sep results to files:
# File for import into positions analyzer:
strg = str(sep)
strg += ' , '
strg += str(sep_stdev)
strg += ' , '
strg += str(pa)
strg += ' , '
strg += str(pa_stdev)

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
string += str(np.mean(xcc))
string += ' , '
string += str(np.mean(ycc)) + "\n"
string += ' star pixel location: '
string += str(np.mean(xcs))
string += ' , '
string += str(np.mean(ycs)) + "\n"
string += ' sep/PA: '
string += str(sep)
string += ' , '
string += str(pa) + "\n"
string += ' sep/PA std devs: '
string += str(sep_stdev)
string += ' , '
string += str(pa_stdev) + "\n"
string += ' RA/Dec: '+ str(RA)+' , ' + str(Dec) + "\n"
string += ' RA/Dec std devs: ' + str(RAstd) +' , ' + str(Decstd) + "\n"
string += ' NSamples: ' + str(minxy-additional_burnin) + "\n"
string += ' Samp mode,Multisam,Coadds,Itime: '
string += str(b) + "\n"
string += ' Vertical angle mode: '
string += a + "\n"
z.write(string + "\n")
z.close()


# Make a triangle plot of parameter results:



# Convert to numpy arrays and flatten:
xcs,ycs,xcc,ycc = np.array(xcs),np.array(ycs),np.array(xcc),np.array(ycc)
amps,ampc,ampratio,bkgd = np.array(amps),np.array(ampc),np.array(ampratio),np.array(bkgd)
sigmax,sigmay,sigmax2,sigmay2,theta,theta2 = np.array(sigmax),np.array(sigmay),np.array(sigmax2),np.array(sigmay2),\
    np.array(theta),np.array(theta2)
xcsf=xcs.flatten()
ycsf=ycs.flatten()
xccf=xcc.flatten()
yccf=ycc.flatten()
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

cornerplot = filename.split('/')[0]+'/'+filename.split('/')[1]+'/'+filename.split('.')[2]+ '_olpefit_results_tacc'

# Make corner plot using "corner":
import corner
minshape = min(xcsf.shape,ycsf.shape,xccf.shape,yccf.shape,ampsf.shape,ampcf.shape,ampratiof.shape,bkgd.shape)
ndim, nsamples = 14, minshape
data = np.vstack([xcsf,ycsf,xccf,yccf,ampsf,ampcf,ampratiof,bkgdf,sigmaxf,sigmayf,sigmax2f,sigmay2f,thetaf,theta2f])
data=data.transpose()
# Plot it.
plt.rcParams['figure.figsize'] = (10.0, 6.0)
figure = corner.corner(data, labels=["xcs", 'ycs', "xcc","ycc",'amps','ampc','ampratio','bkgd','sigmax','sigmay',\
                                    'sigmax2','sigmay2','theta','theta2'],
                       show_titles=True, plot_contours=True, title_kwargs={"fontsize": 12})
figure.savefig(cornerplot+'/'+filename.split('.')[2]+'_cornerplot', dpi=100)

