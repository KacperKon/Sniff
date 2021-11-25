# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:59:04 2021

@author: kkondrakiewicz
"""

# WOT TU DU

# Add docstring to the mean calc functions

# REGRESJA KURWA SNIFFING NA PUPIL

# Przeczytaj uważnie obydwa kody
# Co można z pożytkiem dla swiata zamknąć w funkcje?
# import damych mat
# import danych ze źrenicy
# robienie macierzy valid trials

# To zamknij i załóź repo na gicie

# Jakich wykresów dokładnie potrzebujesz?
    # chyba wszystkich w pętli - każda prezentacja, surowe i znormalizowane itd.
# Czy to źle, że liczysz sniffing w binach na podstawie instantenous?



# Ta ważna czesc:
    
#%% Restructure sniffing data into 3-dim array: trials x time point x mice
# Using similar stracture, calculate breathing rate (multiple methods)
bin_edges = np.arange(0, nframes, binsize*sr)
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
        bsl = np.mean(sniff_gauss[tr, bsln_start*sr:odor_start*sr, m])
        sniff_delta[tr, :, m] = sniff_gauss[tr, :, m] - bsl
        
        sniff_bins[tr,:,m] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
        sniff_delbins[tr,:,m] = sniff_bins[tr,:,m] - sniff_bins[tr,1,m]
        
        sniff_mybin[tr, m] = np.mean(sniff_delta[tr, sniff_the_bin[0]*sr:sniff_the_bin[1]*sr, m])
        
    sniff_list.append(tmp_list) 
    
    
#%%
plt.figure()
m = 0
plt.scatter(sniff_mybin[:, m], pup_mybin[:, m])
plt.figure()
m = 2
plt.scatter(sniff_mybin[:, m], pup_mybin[:, m])
