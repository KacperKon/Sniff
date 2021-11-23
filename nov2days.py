# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:07:17 2021

@author: kkondrakiewicz
"""
from scipy import io
import glob
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy
import sys
import h5py
from scipy.signal import savgol_filter
sys.path.append(r'C:\Users\kkondrakiewicz\Documents\Python Scripts\Sniff')
import utils_sniff as us

#%% Specify paths and some global analysis parameteres

data_path = r'C:\Users\kkondrakiewicz\Desktop\PSAM_SC\data_all'
expect_files = 3 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
pup_nframes = 373 # the same for pupil camera
sigma = 0.25
binsize = 2 # for binned analysis, bin size in seconds
odor_start = 240
odor_end = 360
bsln_start = 120
tvec = np.arange(-4, 7.03, 1/60)
ndays = 2
mybin = [5, 7] # concentrate on this part - from 1 sec to 3 sec after odor presentation

#%% Import sniffing and trial data as a list of dictionaries - 1 dictionary for each mouse or session
all_folders = glob.glob(data_path + '/**/')
all_files = glob.glob(data_path + '/**/*.mat', recursive=True)

nses = len(all_folders) # number of all recording sessions
nmice = int(nses/ndays)

sniffs = []

for mouse in range(nses):
    tmp_dict = {}
    for file in range(expect_files):
        file_idx = mouse*expect_files+file
        tmp_dict = {**tmp_dict, **io.loadmat(all_files[file_idx], squeeze_me=True)}
        
    sniffs.append(tmp_dict)

#%% Exctract some basic info from the imported data
ntrials = sniffs[0]['trial_idx'].size
npres = max(sniffs[0]['trial_occur'])
sr = sniffs[0]['samp_freq']
bin_edges = np.arange(0, nframes, binsize*sr)

#%% Import pupil dilation data
pup_files = glob.glob(data_path + '/**/*.h5', recursive=True)
camlog_files = glob.glob(data_path + '/**/*.camlog', recursive=True)
pup_ts = []
pup_raw = []

for m in range(nses):
    tmp_data = np.array(h5py.File(pup_files[m])['diameter'])
    pup_raw.append(tmp_data)
    
    log,comm = us.parse_cam_log(camlog_files[m])
    pup_ts.append(np.array(log['timestamp']))
    
# Parse pupil data into trials (3-dim array: trials x time point x mice)
pup_m = np.empty([ntrials, pup_nframes, nses])
pup_m[:] = np.nan
pup_delta = pup_m.copy()
pup_mybin = np.zeros([ntrials, nses])

for m in range(nses):
    trial_starts = np.array(np.where(np.diff(pup_ts[m]) > 1e10)[0] + 1)
    trial_starts = np.hstack([np.array([0]), trial_starts])
    for tr in range(ntrials):
        tmp_data = pup_raw[m][trial_starts[tr]:trial_starts[tr]+pup_nframes]
        tmp_data = savgol_filter(tmp_data, 51, 10)
        pup_m[tr, 0:tmp_data.size, m] = tmp_data # sometimes camera drops a few frames in the last trial - this way those will be NaNs
        
        bsl = np.nanmean(tmp_data[2*31:4*31])
        pup_delta[tr, 0:tmp_data.size, m] = (tmp_data - bsl)
        
        pup_mybin[tr, m] = np.nanmean(pup_delta[tr,6*31:8*31,m])
        
        
#%% Restructure sniffing data into 3-dim array: trials x time point x mice
# Using similar stracture, calculate breathing rate (multiple methods)
sniff_ons = np.zeros([ntrials, nframes, nses])
sniff_gauss = sniff_ons.copy()
sniff_delta = sniff_ons.copy()
sniff_list = []
sniff_bins = np.zeros([ntrials, bin_edges.size-1, nses])
sniff_delbins = sniff_bins.copy()
sniff_mybin = np.zeros([ntrials, nses])

for m in range(nses):
    tmp_list = []
    for tr in range(ntrials):
        sniff_ons[tr, sniffs[m]['ml_inh_onsets'][tr], m] = 1 # code sniff onsets as 1
        tmp_list.append(sniffs[m]['ml_inh_onsets'][tr]) # store inhalation onsets also in (list for raster plots)
        
        sniff_gauss[tr, :, m] =  gaussian_filter1d(sniff_ons[tr,:,m], sigma*sr, mode = 'reflect') *sr
        bsl = np.mean(sniff_gauss[tr, bsln_start:odor_start, m])
        sniff_delta[tr, :, m] = sniff_gauss[tr, :, m] - bsl
        
        sniff_bins[tr,:,m] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
        sniff_delbins[tr,:,m] = sniff_bins[tr,:,m] - sniff_bins[tr,1,m]
        
        sniff_mybin[tr, m] = np.mean(sniff_delta[tr, mybin[0]*sr:mybin[1]*sr, m])
        
    sniff_list.append(tmp_list)    
    

#%% Create odor category matrix, indicating for each trial which odor type is it
# Each column corresponds to one category (key) from the 'odor_cat' dictionary

tr_cat = [] # store trial category (fam, novel, blank) in 1 matrix separately for each mouse

for mouse in range(nses):
    trm = np.vstack([sniffs[mouse]['trial_familiarity'], sniffs[mouse]['trial_novelty'], sniffs[mouse]['trial_blank']]).T
    tr_cat.append(trm)
    
ncat = tr_cat[m].shape[1] 
tr_incl = copy.deepcopy(tr_cat) # copy category matrix to impose additional criteria

# Now from the odor category matrix exclude some trials based on presentation no.
for m in range(nses):
    for cat in range(ncat):
        if cat < 1: # for familiar odorants (1 first catumns of odor_cat_m), exclude preblock trials
            valid_trials = np.logical_and(sniffs[m]['trial_occur']>=5, sniffs[m]['trial_occur']<=6)
            valid_trials = np.logical_and(valid_trials, tr_incl[m][:,cat])
        else:
            valid_trials = np.logical_and(tr_incl[m][:,cat], sniffs[m]['trial_occur']<=2)
            #valid_trials = np.logical_and(tr_incl[m][:,cat], sniffs[0]['trial_occur']<=1)
        
        tr_incl[m][:, cat] = valid_trials[:]*1
        
incl_descr = 'First odor presentation'

#%% Calculate mean sniffing across time for selected presentation
sniff_av = np.zeros([nframes, nses, ncat])
sniff_n = np.sum(tr_incl[0], 0)
sniff_sem = sniff_av.copy()

for m in range(nses):
    for cat in range(ncat):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        tmp_data = sniff_delta[which_incl, :, m].T
        sniff_av[:,m,cat] = np.mean(tmp_data, 1)
        sniff_sem[:,m,cat] = np.std(tmp_data, 1) / np.sqrt(sniff_n[cat])
        
        
#%% The same, but for mean pupil
pup_av = np.zeros([pup_nframes, nses, ncat])
pup_n = np.sum(tr_incl[0], 0)
pup_sem = pup_av.copy()

for m in range(nses):
    for cat in range(ncat):
        which_incl = sniffs[m]['trial_idx'][np.where(tr_incl[m][:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        
        tmp_data = pup_delta[which_incl, :, m].T
        
        pup_av[:,m,cat] = np.nanmean(tmp_data, 1)
        pup_sem[:,m,cat] = np.nanstd(tmp_data, 1) / np.sqrt(pup_n[cat])
        
        
#%% Calculate 1 mean sniffing value for all presentations
sniff_1bin_av = np.zeros([npres, nses, ncat])
sniff_1bin_n = sniff_1bin_av.copy()
sniff_1bin_sem = sniff_1bin_av.copy()

for p in range(npres):
    for m in range(nses):
        for cat in range(ncat):
            which_tr = np.logical_and(tr_cat[m][:,cat], sniffs[m]['trial_occur'] == p+1)
            which_rows = sniffs[m]['trial_idx'][which_tr] - 1
            
            tmp_data = sniff_mybin[which_rows, m]
            sniff_1bin_av[p, m, cat] = np.mean(tmp_data)
            sniff_1bin_n[p, m, cat] = tmp_data.size
            sniff_1bin_sem[p, m, cat] = np.std(tmp_data) / np.sqrt(sniff_1bin_n[p, m, cat])
            
            
#%% The same, but for pupil
pup_1bin_av = np.zeros([npres, nses, ncat])
pup_1bin_n = pup_1bin_av.copy()
pup_1bin_sem = pup_1bin_av.copy()

for p in range(npres):
    for m in range(nses):
        for cat in range(ncat):
            which_tr = np.logical_and(tr_cat[m][:,cat], sniffs[m]['trial_occur'] == p+1)
            which_rows = sniffs[m]['trial_idx'][which_tr] - 1
            
            tmp_data = pup_mybin[which_rows, m]
            pup_1bin_av[p, m, cat] = np.nanmean(tmp_data)
            pup_1bin_n[p, m, cat] = tmp_data.size
            pup_1bin_sem[p, m, cat] = np.std(tmp_data) / np.sqrt(pup_1bin_n[p, m, cat])

#%% Plot breathing across time for some selected trials
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all')
axes = axes.flatten()

for m in range(nmice):
    
    axes[m].plot(tvec, sniff_av[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[m].fill_between(tvec, sniff_av[:,m,0] + sniff_sem[:,m,0], sniff_av[:,m,0] - sniff_sem[:,m,0], alpha = 0.2, color = 'C7')
    
    axes[m].plot(tvec, sniff_av[:,m+nmice,0], label = 'FAM DREADD', color = 'C7', linestyle = '--')
    axes[m].fill_between(tvec, sniff_av[:,m+nmice,0] + sniff_sem[:,m+nmice,0], sniff_av[:,m+nmice,0] - sniff_sem[:,m+nmice,0], alpha = 0.2, color = 'C7')

    axes[m].plot(tvec, sniff_av[:,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[m].fill_between(tvec, sniff_av[:,m,1] + sniff_sem[:,m,1], sniff_av[:,m,1] - sniff_sem[:,m,1], alpha = 0.2, color = 'C0')

    axes[m].plot(tvec, sniff_av[:,m+nmice,1], label = 'NOV DREADD', color = 'C0', linestyle = '--')
    axes[m].fill_between(tvec, sniff_av[:,m+nmice,1] + sniff_sem[:,m+nmice,1], sniff_av[:,m+nmice,1] - sniff_sem[:,m+nmice,1], alpha = 0.2, color = 'C0')
    
    axes[m].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
    
    ax2 = axes[m].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    
axes[m].legend()
axes[m].set_xlabel('Time from odor presentation [sec]')
fig.suptitle(incl_descr)

#%% Plot habituation curve for each mouse
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all')
axes = axes.flatten()

pvec_f = np.arange(-3,11)
pvec_n = np.arange(1,11)

for m in range(nmice):
    axes[m].errorbar(pvec_f, sniff_1bin_av[:,m,0], sniff_1bin_sem[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[m].errorbar(pvec_f, sniff_1bin_av[:,m+nmice,0], sniff_1bin_sem[:,m+nmice,0], label = 'FAM DREADD', color = 'C7', linestyle = '--')

    axes[m].errorbar(pvec_n, sniff_1bin_av[:10,m,1], sniff_1bin_sem[:10,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[m].errorbar(pvec_n, sniff_1bin_av[:10,m+nmice,1], sniff_1bin_sem[:10,m+nmice,1], label = 'NOV DREADD', color = 'C0', linestyle = '--')
        
    axes[m].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
    
    ax2 = axes[m].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    
axes[m].legend()
axes[m].set_xticklabels(['PB1', 'PB2', 'PB4', '2', '4', '6', '8', '10'])
axes[m].set_xlabel('Presentation number')
fig.suptitle('Habituation curve')



#%% Plot pupil for selected trials
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all')
axes = axes.flatten()
tvec = np.arange(-4, 8.02, 1/31)

for m in range(nmice):
    
    axes[m].plot(tvec, pup_av[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[m].fill_between(tvec, pup_av[:,m,0] + pup_sem[:,m,0], pup_av[:,m,0] - pup_sem[:,m,0], alpha = 0.2, color = 'C7')
    
    axes[m].plot(tvec, pup_av[:,m+nmice,0], label = 'FAM DREADD', color = 'C7', linestyle = '--')
    axes[m].fill_between(tvec, pup_av[:,m+nmice,0] + pup_sem[:,m+nmice,0], pup_av[:,m+nmice,0] - pup_sem[:,m+nmice,0], alpha = 0.2, color = 'C7')

    axes[m].plot(tvec, pup_av[:,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[m].fill_between(tvec, pup_av[:,m,1] + pup_sem[:,m,1], pup_av[:,m,1] - pup_sem[:,m,1], alpha = 0.2, color = 'C0')

    axes[m].plot(tvec, pup_av[:,m+nmice,1], label = 'NOV DREADD', color = 'C0', linestyle = '--')
    axes[m].fill_between(tvec, pup_av[:,m+nmice,1] + pup_sem[:,m+nmice,1], pup_av[:,m+nmice,1] - pup_sem[:,m+nmice,1], alpha = 0.2, color = 'C0')

    
    axes[m].set_ylabel(u"\u0394" + ' pupil dilation [au]')
    
    ax2 = axes[m].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    
axes[m].legend()
axes[m].set_xlabel('Time from odor presentation [sec]')
fig.suptitle(incl_descr)


#%% Plot pupil habituation curve for each mouse
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all')
axes = axes.flatten()

pvec_f = np.arange(-3,11)
pvec_n = np.arange(1,11)

for m in range(nmice):
    axes[m].errorbar(pvec_f, pup_1bin_av[:,m,0], pup_1bin_sem[:,m,0], label = 'FAM SAL', color = 'C7', linestyle = '-')
    axes[m].errorbar(pvec_f, pup_1bin_av[:,m+nmice,0], pup_1bin_sem[:,m+nmice,0], label = 'FAM DREADD', color = 'C7', linestyle = '--')

    axes[m].errorbar(pvec_n, pup_1bin_av[:10,m,1], pup_1bin_sem[:10,m,1], label = 'NOV SAL', color = 'C0', linestyle = '-')
    axes[m].errorbar(pvec_n, pup_1bin_av[:10,m+nmice,1], pup_1bin_sem[:10,m+nmice,1], label = 'NOV DREADD', color = 'C0', linestyle = '--')
        
    axes[m].set_ylabel(u"\u0394" + ' pupil dilation [au]')
    
    ax2 = axes[m].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
    
    
axes[m].legend()
axes[m].set_xticklabels(['PB1', 'PB2', 'PB4', '2', '4', '6', '8', '10'])
axes[m].set_xlabel('Presentation number')
fig.suptitle('Habituation curve')

