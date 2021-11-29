# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:07:17 2021

@author: kkondrakiewicz
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\kkondrakiewicz\Documents\Python Scripts\Sniff')
import sniff_tools as st

#%% Specify paths and some global analysis parameteres

data_path = r'C:\Users\kkondrakiewicz\Desktop\mixtures'
expect_files = 3 # how many files per mice you expect
nframes = 662 # how many camera frames per trial you expect
binsize = 2 # for binned analysis, bin size in seconds
sigma = 0.25 # for instantenous sniffing rate analysis, sigma of gaussian kernel [sec]
bsln_start = 2 # [sec]
odor_start = 4 # [sec]
odor_end = 6 # [sec]
sniff_the_bin = [5, 7] # the bin of interest for sniffing analysis
preblock = 4

tvec = np.arange(-4, 7.03, 1/60) # time vector for single-trial plotting


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
sniffs = st.import_sniff_mat(data_path)
nmice = len(sniffs)

#%% Exctract some basic info from the imported data
ntrials = sniffs[0]['trial_idx'].size
sr = sniffs[0]['samp_freq']
bin_edges = np.arange(0, nframes, binsize*sr)
    
#%% Restructure sniffing data into 3-dim array: trials x time point x mice
# Using similar stracture, calculate breathing rate (multiple methods)    
sniff_ons, sniff_list, sniff_bins, sniff_delbins, sniff_mybin = st.bin_sniff(sniffs, nframes, bsln_start, odor_start, sniff_the_bin, binsize)
sniff_gauss, sniff_delta = st.ins_sniff(sniff_ons, bsln_start, odor_start, sigma, sr)

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
    
# Now from the odor category matrix exclude some trials based on presentation no.
for cat in range(len(odor_cat)):
    if cat <= 1: # for familiar odorants (2 first catumns of odor_cat_m), exclude preblock trials
        valid_trials = np.logical_and(sniffs[0]['trial_occur']>=5, sniffs[0]['trial_occur']<=5)
        valid_trials = np.logical_and(valid_trials, odor_cat_m[:,cat])
    else:
        valid_trials = np.logical_and(odor_cat_m[:,cat], sniffs[0]['trial_occur']>=1)
        valid_trials = np.logical_and(valid_trials, sniffs[0]['trial_occur']<=1)
    include_tr[valid_trials, cat] = 1

incl_descr = 'First odor presentation'
#incl_descr = 'Second odor presentation'

#%% Calculate for each mouse sniffing time by odor occurence
# because I assumed same trial structure for all mice (odor_cat_m is a matrix, not a list), just repeat it x nmice 
sniff_1bin_av, sniff_1bin_n, sniff_1bin_sem = st.av_by_occur(sniffs, sniff_mybin, nmice*[odor_cat_m]) 
fam_oc = sniff_1bin_av.shape[0]
nov_oc = fam_oc - preblock

#%% Plot breathing across time for each category

fig, axes = plt.subplots(3, 1, sharex = 'col')
axes = axes.flatten()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,cat] == 1)] - 1 # subtract 1, because 'trial_idx' stores Matlab indexes starting from 1 
        tmp_data = sniff_delta[which_to_an, :, m].T
        tmp_data = np.mean(tmp_data, 1)
        axes[m].plot(tvec, tmp_data, label = list(odor_cat.keys())[cat])
        axes[m].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
        
#plt.legend(list(odor_cat.keys()), loc = 'lower right')
plt.suptitle(incl_descr) 
plt.legend()


#%% Based on previous section, calculate average breathing for each odor category

means = np.zeros([nmice, len(odor_cat)])
npres = means.copy()

for m in range(nmice):
    for cat in range(len(odor_cat)):
        which_to_an = sniffs[m]['trial_idx'][np.where(include_tr[:,cat] == 1)] - 1
        means[m, cat] = np.mean(sniff_mybin[which_to_an, m])
        npres[m, cat] = np.sum(include_tr[:,cat])

plt.figure()
for ii in range(len(odor_cat)):
    plt.scatter([ii]*nmice, means[:,ii])
    gr_av = np.mean(means[:,ii])
    gr_sem = np.std(means[:,ii]) / np.sqrt(means[:,ii].size)
    plt.bar(ii, gr_av, yerr = gr_sem, fill=False, color = 'k')
plt.legend(list(odor_cat.keys()))
plt.xticks([])
plt.ylabel(u"\u0394" + ' sniffing [inh/sec]')
plt.title(incl_descr) 
    
#%% Make habituation curves for each animal
fig, axes = plt.subplots(nmice, 1, sharex = 'all', sharey='all')
axes = axes.flatten()

pvec_f = np.arange(-(preblock-1),nov_oc+1) # vector of familiar occurences (preblock <= 0)
pvec_n = np.arange(1,nov_oc+1) # same for novel occurences

for m in range(nmice):
    for cat in range(sniff_1bin_av.shape[2]):
        if cat <= 1:
            axes[m].errorbar(pvec_f, sniff_1bin_av[:,m,cat], sniff_1bin_sem[:,m,cat], label = list(odor_cat.keys())[cat])
        else:
            axes[m].errorbar(pvec_n, sniff_1bin_av[0:nov_oc,m,cat], sniff_1bin_sem[0:nov_oc,m,cat], label = list(odor_cat.keys())[cat])
            
            axes[m].set_ylabel(u"\u0394" + ' sniffing [inh/sec]')
    
    ax2 = axes[m].twinx()
    ax2.set_yticks([])
    mouse_id = sniffs[m]['unique_id'][7:12]
    ax2.set_ylabel('Mouse ' + mouse_id)
     
axes[m].legend()
axes[m].set_xticklabels(['PB1', 'PB2', 'PB4', '2', '4', '6', '8'])
axes[m].set_xlabel('Presentation number')
fig.suptitle('Habituation curve - individual mice')


#%% And one average hab curve
hab_av = np.nanmean(sniff_1bin_av,1)
hab_av[4:12,2:] = hab_av[0:8,2:]
hab_av[0:4,2:] = np.nan

hab_sem = np.nanstd(sniff_1bin_av,1) / np.sqrt(nmice)
hab_sem[4:12,2:] = hab_sem[0:8,2:]
hab_sem[0:4,2:] = np.nan

plt.figure()
for cat in range(len(odor_cat)):
    plt.errorbar(pvec_f, hab_av[:,cat], hab_sem[:,cat])
    
plt.xticks(np.arange(-2, 9, 2), labels = ['PB2', 'PB4', '2', '4', '6', '8'])
plt.title('Habituation curve - average')
plt.ylabel(u"\u0394" + ' sniffing [inh/sec]')
plt.legend(list(odor_cat.keys()))



    
