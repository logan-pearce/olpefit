# This code executes OLPE fit step 4 for an entire epoch at once

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
    print 'Processing '+ line.split('.')[2]
    filename = line
    os.system('python olpe_fit_tacc_step4.py '+line)

os.system('rm filelist')
