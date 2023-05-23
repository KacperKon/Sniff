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
import matplotlib as mpl

#%% Specify paths and some global analysis parameteres
sniff_path = r'C:\Users\Kacper\Desktop\preBotz\data'
sess_ids = ['230313_KK045', '230313_KK046', '230315_KK045', '230315_KK046']
#sess_ids = ['230309_KK045', '230309_KK046']
#sess_ids = ['230316_KK045', '230316_KK046']  
 
fig_path = r'C:\Users\Kacper\Desktop\preBotz\no_odors'


#expect_files = 4 # how many files per mice you expect
nframes = 722 # how many camera frames per trial you expect
pup_nframes = 373 # the same for pupil camera
pup_sr = pup_nframes/12
sigma = 0.25
binsize = 0.5 # for binned analysis, bin size in seconds
est_lat = 0.2 # estimated olfactometer latency
odor_start = 4 + est_lat
odor_end = 6 + est_lat
bsln_start = 1
ndays = 1
sniff_the_bin = [4.2, 8.2]
figure_size = (8, 9)
mpl.rcParams['svg.fonttype'] = 'none' # this should make text editable in .svg figures

var_color = ['blue', 'red']
var_label = ['Inhibition', 'Stimulation']

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

#%% Restructure sniffing data into 3-dim array: trials x time point x miceand calculate breathing rate (multiple methods)
sniff_ons, sniff_list, sniff_bins, sniff_delbins, sniff_mybin = st.bin_sniff(sniffs, \
        nframes, bsln_start, odor_start, sniff_the_bin, binsize, classic_tracker=False)
sniff_gauss, sniff_delta = st.ins_sniff(sniff_ons, bsln_start, odor_start, sigma, sr)


#%%
tvec = np.linspace(-4, 8, nframes) + est_lat


for m in range(nses):
    
    if min(sniffs[m]['trial_opto']) == -1:
        ii = 0
    elif max(sniffs[m]['trial_opto']) == 1:
        ii = 1
    
    n_freqs = len(np.unique(sniffs[m]['trial_led_freq']))
    fig, axes = plt.subplots(n_freqs, 2, sharex = 'all', sharey='col', figsize = figure_size, dpi = 250)
    fig.subplots_adjust(top=0.95)
    
    plt.figure(figsize = (4,4))
    ax_scat = plt.axes()
    

    for row, fs in enumerate(np.unique(sniffs[m]['trial_led_freq'])):
        
        which_rows = sniffs[m]['trial_led_freq'] == fs
        which_tr = sniffs[m]['trial_idx'][which_rows] - 1
        
        led_tmp = [sniffs[m]['led_ons'][i] - (4-est_lat) for i in which_tr]
        inh_tmp = [sniffs[m]['ft_inh_onsets'][i]/sr - (4-est_lat) for i in which_tr]
        
        axes[row,0].eventplot(inh_tmp, color = 'gray')
        #axes[row,0].eventplot(led_tmp)

        axes[row,0].axvline(0, color = var_color[ii], alpha = 0.5)
        axes[row,0].axvline(4, color = var_color[ii], alpha = 0.5)
        
        axes[row,1].plot(tvec, sniff_gauss[which_tr, :, m].T, color='gray', alpha = 0.5)
        axes[row,1].plot(tvec, np.mean(sniff_gauss[which_tr, :, m],0), color='black')
        axes[row,1].axvline(0, color = var_color[ii], alpha = 0.5)
        axes[row,1].axvline(4, color = var_color[ii], alpha = 0.5)
        
        axes[row,1].set_xlim([-3, 7])
        #axes[row,1].set_ylim([-0.3, 10])
        
        axes2 = axes[row, 1].twinx()
        axes2.set_yticks([])
        if fs == 0:
            right_label = 'Control'
        else:
            right_label = str(fs) + ' Hz'
        axes2.set_ylabel(right_label)
        
        axes[row,0].set_ylabel('Trial')
        axes[row,1].set_ylabel('Inh/sec')
        
        av_tmp = np.mean(sniff_mybin[which_tr, m])
        ax_scat.scatter(fs, av_tmp, color = 'black')

    title = var_label[ii] + '-' + sniffs[m]['unique_id']
    fig.suptitle(title, weight = 'bold')   
     
    axes[row,0].set_xlabel('Time from stim [sec]')
    axes[row,1].set_xlabel('Time from stim [sec]')
    
    fig.savefig(fig_path + '\\' + title + '.png', transparent = True)
    fig.savefig(fig_path + '\\' + title + '.svg', bbox_inches = 'tight', transparent = True)

    
    ax_scat.set_ylim([0,7])
    ax_scat.set_title(title)
    ax_scat.set_ylabel('Inhalations / sec', fontsize = 14)
    ax_scat.set_xlabel('Light pulses frequency', fontsize = 14)
    plt.savefig(fig_path + '\\' + 'freqs_' + title + '.png', bbox_inches = 'tight', dpi = 250, transparent = True)
    plt.savefig(fig_path + '\\' + 'freqs_' + title + '.svg', bbox_inches = 'tight', transparent = True)



