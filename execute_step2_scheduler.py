# This script executes the command line code to schedule Lonestar5 TACC jobs for all images in an epoch for step 2.

import os
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
    filename = line
    epoch = filename.split('/')[1]
    os.system('sbatch '+directory+'sbatch/'+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'_sbatch_step2')

os.system('rm filelist')


