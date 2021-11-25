# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:36:45 2021
@author: kkondrakiewicz

Functions for analyzing breathing rate and pupil diameter from head-fixed animals.
"""
import re
import pandas as pd
import glob
from scipy import io
import numpy as np
import h5py
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

#%% Functions for behavioral (sniffing) data

def import_sniff_mat(data_path, expect_files = 3):
    r"""
    Import pre-processed behavioral data on sniffing and trial structure.
    Assumes the 

    Parameters
    ----------
    data_path : string, eg. r'C:\Users\Freud'
    expect_files : how many .mat files there should be for 1 recording session

    Returns
    -------
    sniffs : a list of dictionaries 
    Should contain all the fields from imported Matlab structures (in a single dictionary for each mouse / session).
    
    """
    
    all_folders = glob.glob(data_path + '/**/')
    all_files = glob.glob(data_path + '/**/*.mat', recursive=True)
    
    nses = len(all_folders) # number of all recording sessions
    
    sniffs = []
    
    for mouse in range(nses):
        tmp_dict = {}
        for file in range(expect_files):
            file_idx = mouse*expect_files+file
            tmp_dict = {**tmp_dict, **io.loadmat(all_files[file_idx], squeeze_me=True)}
            
        sniffs.append(tmp_dict)
        
    return sniffs


def bin_sniff(sniffs, nframes, bsln_start, odor_start, the_bin, binsize):

    nses = len(sniffs)
    ntrials = sniffs[0]['trial_idx'].size
    sr = sniffs[0]['samp_freq']
    bin_edges = np.arange(0, nframes, binsize*sr)
    
    sniff_ons = np.zeros([ntrials, nframes, nses])
    sniff_list = []
    sniff_hist = np.zeros([ntrials, bin_edges.size-1, nses])
    sniff_delhist = sniff_hist.copy()
    sniff_mybin = np.zeros([ntrials, nses])

    for m in range(nses):
        tmp_list = []
        for tr in range(ntrials):
            sniff_ons[tr, sniffs[m]['ml_inh_onsets'][tr], m] = 1 # code sniff onsets as 1
            tmp_list.append(sniffs[m]['ml_inh_onsets'][tr]) # store inhalation onsets also in (list for raster plots)
                
            sniff_hist[tr,:,m] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
            bsln = np.sum(sniff_ons[tr, int(bsln_start*sr) : int(odor_start*sr), m])/binsize
            sniff_delhist[tr,:,m] = sniff_hist[tr,:,m] - bsln
                
            sniff_mybin[tr, m] = np.sum(sniff_ons[tr, the_bin[0]*sr:the_bin[1]*sr, m])/(the_bin[1] - the_bin[0]) - bsln
                
        sniff_list.append(tmp_list)
        
    return sniff_ons, sniff_list, sniff_hist, sniff_delhist, sniff_mybin


def ins_sniff(sniff_ons, bsln_start, odor_start, sigma, sr = 60):
    sniff_gauss = np.zeros(sniff_ons.shape)
    sniff_delta = np.zeros(sniff_ons.shape)
    
    nses = sniff_ons.shape[2]
    ntrials = sniff_ons.shape[0]
    
    for m in range(nses):
        for tr in range(ntrials):
            sniff_gauss[tr, :, m] =  gaussian_filter1d(sniff_ons[tr,:,m], sigma*sr, mode = 'reflect') *sr
            bsl = np.mean(sniff_gauss[tr, int(bsln_start*sr):int(odor_start*sr), m])
            sniff_delta[tr, :, m] = sniff_gauss[tr, :, m] - bsl
     
    return sniff_gauss, sniff_delta
    

#%% Functions for face camera (pupil) data

def import_pupil(data_path, file_ext = 'h5'):
    r"""
    Import data on pupil dilation, as processed by mptracker. 
    
    Parameters
    ----------
    data_path : string, eg. r'C:\Users\Freud'
    file_ext : string - extension used for saving the pupil dilation hdf5 files. The default is 'h5'.
        The timestamp data is always saved with .camlog extension.

    Returns
    -------
    pup_raw : list of arrays - pupil diameter for each frame (x n sessions/mice)
    NOTE: can be easily extended to import also other data, like pupil position
    pup_ts : list of arrays - timestamp of each frame with pupil data  (x n sessions/mice)

    """
    
    pup_files = glob.glob(data_path + '/**/*.' + file_ext, recursive=True)
    camlog_files = glob.glob(data_path + '/**/*.camlog', recursive=True)
    pup_ts = []
    pup_raw = []
    
    for m in range(len(pup_files)):
        tmp_data = np.array(h5py.File(pup_files[m])['diameter'])
        pup_raw.append(tmp_data)
        
        log,comm = parse_cam_log(camlog_files[m])
        pup_ts.append(np.array(log['timestamp']))
        
    return pup_raw, pup_ts


def parse_pupil(pup_raw, pup_ts, ntrials, nframes, nses, smoothen = 0, window_length = 51, polyorder = 10):
    r"""
    Take raw pupil data and parse it into an array trials x frames x sessions
    Optionally, smoothen the data with Savitzky-Golai filter (scipy.signal.savgol_filter)
    
    Parameters
    ----------
    pup_raw : output of import_pupil
    pup_ts :  output of import_pupil
    ntrials, nframes, nses : integers, desired output dimensions 
    smoothen : integer - if 1, the filter will be apploed. The default is 0.
    window_length : integer - filter parameter. The default is 51.
    polyorder : integer - filter parameter. The default is 10.

    Returns
    -------
    pup_m : numpy array with restructured data

    """
    
    pup_m = np.empty([ntrials, nframes, nses])
    pup_m[:] = np.nan
    
    for m in range(nses):
        trial_starts = np.array(np.where(np.diff(pup_ts[m]) > 1e10)[0] + 1)
        trial_starts = np.hstack([np.array([0]), trial_starts])
        for tr in range(ntrials):
            tmp_data = pup_raw[m][trial_starts[tr]:trial_starts[tr]+nframes]
            if smoothen == 1:
                tmp_data = savgol_filter(tmp_data, window_length, polyorder)
            pup_m[tr, 0:tmp_data.size, m] = tmp_data # sometimes camera drops a few frames in the last trial - this way those will be NaNs
            
    return pup_m
        

# Copied from https://github.com/jcouto/labcams/blob/main/notebooks/parse_log_comments.ipynb
def parse_cam_log(fname):
    logheaderkey = '# Log header:'
    comments = []
    with open(fname,'r') as fd:
        for line in fd:
            if line.startswith('#'):
                line = line.strip('\n').strip('\r')
                comments.append(line)
                if line.startswith(logheaderkey):
                    columns = line.strip(logheaderkey).strip(' ').split(',')

    logdata = pd.read_csv(fname,names = columns, 
                          delimiter=',',
                          header=None,
                          comment='#',
                          engine='c')
    return logdata,comments

# Copied from https://github.com/jcouto/labcams/blob/main/notebooks/parse_log_comments.ipynb
def parse_log_comments(comm,msg = 'trial_start:',strkeyname = 'i_trial',strformat=int):
    # this will get the start frames for each trial
    comm = map(lambda x: x.strip('#').replace(' ',''),comm)
    commmsg = list(filter(lambda x:msg in x ,comm))
    table = []
    for i,lg in enumerate(commmsg):
        lg = re.split(',|-|'+msg,lg)
        table.append({strkeyname:strformat(lg[-1]),
                         'iframe':int(lg[0]),
                         'timestamp':float(lg[1])})
    return pd.DataFrame(table)