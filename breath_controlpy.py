# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:08:22 2023

@author: Kacper
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append(r'C:\Users\Kacper\Documents\Python Scripts\Sniff')
import sniff_tools as st

#%% Specify paths and some global analysis parameteres
sniff_path = r'C:\Users\Kacper\Desktop\LDTg_Chat_BIPOLES\data_all'
#sess_ids = ['221118_KK036', '221118_KK037', '230225_KK042', '230225_KK043', '230225_KK044']
sess_ids = ['221114_KK036', '221114_KK037', '230221_KK042', '230221_KK043', '230221_KK044']  
 
fig_path = r'C:\Users\Kacper\Desktop\LDTg_Chat_BIPOLES\stim2_all'


#expect_files = 4 # how many files per mice you expect
nframes = 722 # how many camera frames per trial you expect
pup_nframes = 373 # the same for pupil camera
pup_sr = pup_nframes/12
sigma = 0.25
binsize = 2 # for binned analysis, bin size in seconds
est_lat = 0 # estimated olfactometer latency
odor_start = 4 + est_lat
odor_end = 6 + est_lat
bsln_start = 1
ndays = 1
sniff_the_bin = [4.5, 7.5] # concentrate on this part - from 1 sec to 3 sec after odor presentation
pup_bin = [5.5, 8.5] # this can be different for pupil, which has slower dynamics


nov_color = '#006400'
fam_color = '#580F41'

sniff_dirs = []
for ses in sess_ids:
    tmp = sniff_path + '\\' + ses
    sniff_dirs.append(tmp)

#%% Import odor trial data
sniffs = st.import_sniff_mat_select(sniff_dirs)


#%% Exctract some basic info from the imported data
nses = len(sniffs)
nmice = int(nses/ndays)
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])
sr = sniffs[0]['samp_freq']
figure_size = (6, nmice*3)
