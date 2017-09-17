## Use this script to upload all images in an observation epoch to TACC work directory

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder",type=str)
args = parser.parse_args()
folder=args.folder
directory = folder.split('/')[0]+'/'+folder.split('/')[1]+'/'
#  Make a list of all images in an epoch:
os.system('ls '+directory+'*.fits > filelist')

with open('filelist') as f:
    z = f.read().splitlines()
# Upload images one at a time:

for line in z:
    os.system('scp '+directory+line+' lap2756@ls5.tacc.utexas.edu:/work/05030/lap2756/lonestar/'+directory
