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

data_path = r'C:\Users\kkondrakiewicz\Desktop\mixtures'
expect_files = 3 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
sigma = 0.25
binsize = 2 # for binned analysis, bin size in seconds
odor_start = 240
odor_end = 360
bsln_start = 120
tvec = np.arange(-4, 7.03, 1/60)

#%% Group odorants by categories
odor_cat = {'fam_sing': [102, 107, 108, 113, 124, 104],
            'fam_mix': [200, 201],
            '1st_tog': [202, 203],
            '1st_alone': [105, 121],
            'nov_sing': [114, 109],
            'nov+fam': [204, 205],
            'nov+nov': [206, 207],
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
odor_delays = []
for m in range(nmice):
    cam_tdms = sniffs[m]['tdms_events'][np.arange(0, ntrials, 4)]
    odor_tdms = sniffs[m]['tdms_events'][np.arange(1, ntrials, 4)]
    odor_delays.append(odor_tdms - cam_tdms)
    
plt.figure()
plt.hist(odor_delays)
    
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
    
#%% Now from the odor category matrix exclude some trials based on presentation no.

for cat in range(len(odor_cat)):
    if cat <= 1: # for familiar odorants (2 first catumns of odor_cat_m), exclude preblock trials
        valid_trials = np.logical_and(sniffs[0]['trial_occur']>=5, sniffs[0]['trial_occur']<=5)
        valid_trials = np.logical_and(valid_trials, odor_cat_m[:,cat])
    else:
        valid_trials = np.logical_and(odor_cat_m[:,cat], sniffs[0]['trial_occur']<=1)
        
    include_tr[valid_trials, cat] = 1

#%% Based on previous section, calculate average breathing for each odor category

means = np.zeros([nmice, len(odor_cat)])
npres = means.copy()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        means[m, cat] = np.mean(sniff_delbins[np.where(include_tr[:,cat]==1),2, m])
        npres[m, cat] = np.sum(include_tr[:,cat])

plt.figure()
for ii in range(len(odor_cat)):
    plt.scatter([ii, ii, ii], means[:,ii])
    plt.legend(list(odor_cat.keys()))

#%% Plot breathing across time for each category

fig, axes = plt.subplots(3, 1, sharex = 'col')
axes = axes.flatten()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,cat] == 1)] - 1
        tmp_data = sniff_delta[which_to_an, :, m].T
        tmp_data = np.mean(tmp_data, 1)
        axes[m].plot(tvec, tmp_data, label = list(odor_cat.keys())[cat])
        
#plt.legend(list(odor_cat.keys()), loc = 'lower right')
plt.legend()











#%%
mean_nov = np.zeros([nmice, nframes])
mean_fam = np.zeros([nmice, nframes])
# select first 4 occurences of new odorants
#which_nov = np.squeeze(sniffs[0]['trial_novelty'] == 1) & np.squeeze(sniffs[0]['trial_occur'] <= 4)
which_nov = (sniffs[0]['trial_chem_id'] == 114) & (sniffs[0]['trial_occur'] <= 4)
# and first 4 of familiar ones (preblock)
which_fam = (sniffs[0]['trial_familiarity'] == 1) & (sniffs[0]['trial_occur'] > 4) & (sniffs[0]['trial_occur'] <=8)


for m in range(nmice):
    mean_nov[m,:] = np.mean(sniff_delta[which_nov,:,m], 0)
    mean_fam[m,:] = np.mean(sniff_delta[which_fam,:,m], 0)


#%%
mouse = 2
plt.figure()
plt.plot(mean_nov[mouse,:].T, 'g')
plt.plot(mean_fam[mouse,:].T, 'k')

#%%
for m in range(nmice):
    plt.figure()
    plt.eventplot(sniff_list[m])
    plt.axvline(odor_start, linestyle = '--', color = 'gray', linewidth = 1)
    plt.axvline(odor_end, linestyle = '--', color = 'gray', linewidth = 1)
    
#%%
plt.figure()
m = 0
tr = 20
plt.plot(np.repeat(sniff_bins[tr,:,m], binsize*sr))
plt.plot(np.repeat(sniff_delbins[tr,:,m], binsize*sr))
plt.plot(sniff_gauss[tr,:,m])
plt.eventplot(sniff_list[m][tr])



# 102 in odor_cat['familiar single']
    
