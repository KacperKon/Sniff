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

#%% Specify paths and some global analysis parameteres

data_path = r'C:\Users\kkondrakiewicz\Desktop\PSAM_SC\data2'
expect_files = 3 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
sigma = 0.25
binsize = 2 # for binned analysis, bin size in seconds
odor_start = 240
odor_end = 360
bsln_start = 120
tvec = np.arange(-4, 7.03, 1/60)

#%% Group odorants by categories
odor_cat = {'familiar': [102, 107, 105, 121],
            #'novel': [108, 113, 124, 104, 114, 109, 123, 120],
            'novel': [106, 118, 100, 103, 111, 112, 119, 115],
            'blank': [117]}

#%% Import data as a list of dictionaries - 1 dictionary for each mouse or session
all_folders = glob.glob(data_path + '/**/')
all_files = glob.glob(data_path + '/**/*.mat', recursive=True)

nmice = len(all_folders)

sniffs = []

for mouse in range(nmice):
    tmp_dict = {}
    for file in range(expect_files):
        file_idx = mouse*expect_files+file
        tmp_dict = {**tmp_dict, **io.loadmat(all_files[file_idx], squeeze_me=True)}
        
    sniffs.append(tmp_dict)
    
#%% Exctract some basic info from the imported data
ntrials = sniffs[0]['trial_idx'].size
sr = sniffs[0]['samp_freq']
bin_edges = np.arange(0, nframes, binsize*sr)

#%% Verify delay between camera starts and odor presentations
# =============================================================================
# odor_delays = []
# for m in range(nmice):
#     cam_tdms = sniffs[m]['tdms_events'][np.arange(0, ntrials, 4)]
#     odor_tdms = sniffs[m]['tdms_events'][np.arange(1, ntrials, 4)]
#     odor_delays.append(odor_tdms - cam_tdms)
#     
# plt.figure()
# plt.hist(odor_delays)
#     
# =============================================================================
#%% Restructure sniffing data into 3-dim array: trials x time point x mice
# Using similar stracture, calculate breathing rate (multiple methods)
sniff_ons = np.zeros([ntrials, nframes, nmice])
sniff_gauss = sniff_ons.copy()
sniff_delta = sniff_ons.copy()
sniff_list = []
sniff_bins = np.zeros([ntrials, bin_edges.size-1, nmice])
sniff_delbins = sniff_bins.copy()

for m in range(nmice):
    tmp_list = []
    for tr in range(ntrials):
        sniff_ons[tr, sniffs[m]['ml_inh_onsets'][tr], m] = 1 # code sniff onsets as 1
        tmp_list.append(sniffs[m]['ml_inh_onsets'][tr]) # store inhalation onsets also in (list for raster plots)
        
        sniff_gauss[tr, :, m] =  gaussian_filter1d(sniff_ons[tr,:,m], sigma*sr, mode = 'reflect') *sr
        bsl = np.mean(sniff_gauss[tr, bsln_start:odor_start, m])
        sniff_delta[tr, :, m] = sniff_gauss[tr, :, m] - bsl
        
        sniff_bins[tr,:,m] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
        sniff_delbins[tr,:,m] = sniff_bins[tr,:,m] - sniff_bins[tr,1,m]
        
    sniff_list.append(tmp_list)    
    

#%% Create odor category matrix, indicating for each trial which odor type is it
# Each column corresponds to one category (key) from the 'odor_cat' dictionary
odor_cat_m = np.zeros([ntrials, len(odor_cat)], int)
include_tr = odor_cat_m.copy()

key_idx = 0
for key in odor_cat.keys():
    for tr in range(ntrials):
        if sniffs[0]['trial_chem_id'][tr] in odor_cat[key]:
            odor_cat_m[tr, key_idx] = 1
    key_idx = key_idx + 1
    
# IF IT'S NOT FAM NOR BLANK, IT'S NOVEL
#odor_cat_m[np.sum(odor_cat_m, 1)==0, 1] = 1  
    
# Now from the odor category matrix exclude some trials based on presentation no.

for cat in range(len(odor_cat)):
    if cat < 1: # for familiar odorants (1 first catumns of odor_cat_m), exclude preblock trials
        valid_trials = np.logical_and(sniffs[0]['trial_occur']>=5, sniffs[0]['trial_occur']<=6)
        valid_trials = np.logical_and(valid_trials, odor_cat_m[:,cat])
    else:
        valid_trials = np.logical_and(odor_cat_m[:,cat], sniffs[0]['trial_occur']<=2)
        #valid_trials = np.logical_and(valid_trials, sniffs[0]['trial_occur']<=4)
        
    include_tr[valid_trials, cat] = 1


#%% Based on previous section, calculate average breathing for each odor category

means = np.zeros([nmice, len(odor_cat)])
npres = means.copy()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,cat] == 1)] - 1
        means[m, cat] = np.mean(sniff_delbins[which_to_an,2, m])
        npres[m, cat] = np.sum(include_tr[:,cat])

plt.figure()
for ii in range(len(odor_cat)):
    plt.scatter([ii, ii], means[:,ii])
    plt.legend(list(odor_cat.keys()))

#%% Plot breathing across time for each category

fig, axes = plt.subplots(2, 1, sharex = 'col')
axes = axes.flatten()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,cat] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
        # which_to_an = np.where(include_tr[:,cat] == 1)
        tmp_data = np.squeeze(sniff_gauss[which_to_an, :, m]).T
        tmp_data = np.mean(tmp_data, 1)
        axes[m].plot(tvec, tmp_data, label = list(odor_cat.keys())[cat])
        axes[m].set_ylim([-2, 8])
        #axes[m].plot(tvec, tmp_data, label = cat)
        
# plt.legend(list(odor_cat.keys()), loc = 'lower right')
#bz = list(odor_cat.keys())[::-1]
#plt.legend(bz, loc = 'lower right')
plt.legend()


#%% Plot breathing across time for each category
# =============================================================================
# fig, axes = plt.subplots(2, 1, sharex = 'col')
# axes = axes.flatten()
# 
# for m in range(nmice):
#     which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,0] == 1)] - 1 # IN MATLAB TRIAL INDEXES START FROM 1!!
#     tmp_data = np.squeeze(sniff_delta[which_to_an, :, m]).T
#     #tmp_data = np.mean(tmp_data, 1)
#     axes[m].plot(tvec, tmp_data)
#     axes[m].set_ylim([-4, 10])
# =============================================================================

#%%
for m in range(nmice):
    plt.figure()
    plt.eventplot(sniff_list[m])
    plt.axvline(odor_start, linestyle = '--', color = 'gray', linewidth = 1)
    plt.axvline(odor_end, linestyle = '--', color = 'gray', linewidth = 1)