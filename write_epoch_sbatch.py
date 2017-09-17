### This script takes in an observation epoch and outputs a an sbatch text file for scheduling both step 2 and step 3 jobs on TACC for each image in the epoch.  It just writes the files, which then need to be scp'd to TACC work directory to be run.  It schedules step 2 to run for 6 hours and step 3 to run for 24 hours.
# The command line to execute thsi script is: python write_epoch_sbatch.py path to obs epoch
# example: python write_epoch_sbatch.py GSC6214/2017_06_27/

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder",type=str)
args = parser.parse_args()
folder=args.folder
directory = folder.split('/')[0]+'/'+folder.split('/')[1]+'/'

os.system('ls '+directory+'*.fits > filelist')
exists = os.path.exists(directory+'sbatch/')
if exists:
    pass
else:
    os.system('mkdir '+directory+'sbatch/')

with open('filelist') as f:
    z = f.read().splitlines()

for line in z:
    filename = line
    epoch = filename.split('/')[1]
    b = open(directory+'/sbatch/'+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'_sbatch_step2', 'w')
    b.write('#!/bin/bash'+"\n"\
'#SBATCH -J '+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'         # job name'+"\n"\
'#SBATCH -o '+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'.o%j     # output and error file name (%j expands to jobID)'+"\n"\
'#SBATCH -N 1			      # number of nodes requested'+"\n"\
'#SBATCH -n 1                          # total number of mpi tasks requested'+"\n"\
'#SBATCH -p normal                     # queue (partition) -- normal, development, etc.'+"\n"\
'#SBATCH -t 6:00:00                   # run time (hh:mm:ss) - 1.5 hours, this takes ~30h'+"\n"\
'#SBATCH --mail-user=loganpearce55@gmail.com'+"\n"\
'#SBATCH --mail-type=begin'+"\n"\
'#SBATCH --mail-type=end               # email me when the job finishes'+"\n"\
"\n"\
'python olpe_fit_tacc_step2.py '+line)
    b.close()
    
    b = open(directory+'/sbatch/'+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'_sbatch_step3', 'w')
    b.write('#!/bin/bash'+"\n"\
'#SBATCH -J '+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'         # job name'+"\n"\
'#SBATCH -o '+filename.split('/')[0]+'_'+epoch.split('_')[0]+'_'+filename.split('.')[2]+'.o%j     # output and error file name (%j expands to jobID)'+"\n"\
'#SBATCH -N 1			      # number of nodes requested'+"\n"\
'#SBATCH -n 1                          # total number of mpi tasks requested'+"\n"\
'#SBATCH -p normal                     # queue (partition) -- normal, development, etc.'+"\n"\
'#SBATCH -t 24:00:00                   # run time (hh:mm:ss) - 1.5 hours, this takes ~30h'+"\n"\
'#SBATCH --mail-user=loganpearce55@gmail.com'+"\n"\
'#SBATCH --mail-type=begin'+"\n"\
'#SBATCH --mail-type=end               # email me when the job finishes'+"\n"\
"\n"\
'python olpe_fit_tacc_step3.py '+line)
    b.close()
    

os.system('rm filelist')
