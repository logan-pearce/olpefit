# This script downloads all the MCMC result files from TACC for an entire epoch
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder",type=str)
args = parser.parse_args()
folder=args.folder
directory = folder.split('/')[0]+'/'+folder.split('/')[1]+'/'

os.system('ls '+directory+'*.fits > filelist')

with open('filelist') as f:
    z = f.read().splitlines()

for line in z:
    exists = os.path.exists(directory+line.split('.')[2]+ '_olpefit_results_tacc')
    if exists:
        pass
    else:
        os.system('mkdir '+directory+line.split('.')[2]+ '_olpefit_results_tacc')
        print 'Making directory for ', line

### Download MCMC files from TACC
for line in z:
    taccfolder = directory+line.split('.')[2]+ '_olpefit_results'
    os.system('scp lap2756@ls5.tacc.utexas.edu:/work/05030/lap2756/lonestar/'+taccfolder+'/* '+\
              directory+line.split('.')[2]+ '_olpefit_results_tacc')
    print 'Downloading files from ',line

os.system('rm filelist')







    
