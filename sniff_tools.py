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
import copy

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
    r"""
    Based on imported .mat strutcures, calculate sniffing rate in bins.

    Parameters
    ----------
    sniffs : output of the import_sniff_mat function
    nframes : how many frames you expect in each trial from the sniff camera
    bsln_start : start of baseline for breathing rate normalization [sec]
    odor_start : [sec] - it's also end of baseline period
    the_bin : list of 2 elements [sec] - edges of your bin of interest - size of this bin might be different from the binsize
    binsize : [sec] - size of bin for calculation of breatinh rate

    Returns
    -------
    sniff_ons : 3-dim array (n trials x n frames x n mice/sessions) with binarized sniff onsets (onset == 1) 
    sniff_list : the same data as above, but stored as list of lists (mice >> trials) - handy for rasterplots
    sniff_hist : 3-dim array (n trials x n bins x n mice/sessions) 
                a histogram of breaths count for each bin, normalized by the bin size (so in Hz) 
    sniff_delhist : as above, but the breathing rate from the baseline period is subtracted
    sniff_mybin : 2-dim array (n trials x n mice/sessions) - sniffing rate for one selected bin, which can be of different size 
                then the 'binsize', subtracted from the baseline

    """

    nses = len(sniffs)
    ntrials = sniffs[0]['trial_idx'].size
    sr = sniffs[0]['samp_freq']
    bin_edges = np.arange(0, nframes, binsize*sr)
    bsln_size = (int(odor_start*sr) - int(bsln_start*sr)) / sr
    
    sniff_ons = np.zeros([ntrials, nframes, nses])
    sniff_list = []
    sniff_hist = np.zeros([ntrials, bin_edges.size-1, nses])
    sniff_delhist = sniff_hist.copy()
    sniff_mybin = np.zeros([ntrials, nses])

    for m in range(nses):
        tmp_list = []
        for tr in range(ntrials):
            sniff_ons[tr, sniffs[m]['ml_inh_onsets'][tr], m] = 1 # code sniff onsets as 1
            tmp_list.append(sniffs[m]['ml_inh_onsets'][tr]) # store inhalation onsets also in a list (for raster plots)
                
            sniff_hist[tr,:,m] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
            bsln = np.sum(sniff_ons[tr, int(bsln_start*sr) : int(odor_start*sr), m])/bsln_size
            sniff_delhist[tr,:,m] = sniff_hist[tr,:,m] - bsln
                
            sniff_mybin[tr, m] = np.sum(sniff_ons[tr, int(the_bin[0]*sr) : int(the_bin[1]*sr), m])/(the_bin[1] - the_bin[0]) - bsln
                
        sniff_list.append(tmp_list)
        
    return sniff_ons, sniff_list, sniff_hist, sniff_delhist, sniff_mybin


def ins_sniff(sniff_ons, bsln_start, odor_start, sigma = 0.25, sr = 60):
    r"""
    Based on sniff onsets, calculate instantaneous breathing rate.
    
    Watch out: some boundary effects might be visible at the start and end of each trial. 
    The gaussian_filter1d function currently tries to deal with it by interpolation (mode = 'reflect'). 

    Parameters
    ----------
    sniff_ons : output of 'bin_sniff' function - array with binarized sniff onsets
    bsln_start : start of baseline period for normalization [sec]
    odor_start : end of baseline period for normalization [sec]
    sigma : width of Gaussian kernel [sec]
    sr : sampling rate of your data, default the same as sniff camera (60fps)

    Returns
    -------
    sniff_gauss : 3-dim array (n trials x n frames x n mice/sessions) with instantenous sniffing rate
    sniff_delta : as above, but with baseline sniffing subtracted

    """
    
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


def select_trials_nov(sniffs, fam_min, fam_max, nov_min, nov_max):
    r"""
    Create a boolean array for deciding which trials should be used based on category & presentation number.

    Parameters
    ----------
    sniffs : TYPE
        DESCRIPTION.
    fam_min : TYPE
        DESCRIPTION.
    fam_max : TYPE
        DESCRIPTION.
    nov_min : TYPE
        DESCRIPTION.
    nov_max : TYPE
        DESCRIPTION.

    Returns
    -------
    tr_cat : TYPE
        DESCRIPTION.
    tr_incl : TYPE
        DESCRIPTION.

    """
    tr_cat = [] # store trial category (fam, novel, blank) in 1 matrix separately for each mouse
    
    nses = len(sniffs)
    for mouse in range(nses):
        trm = np.vstack([sniffs[mouse]['trial_familiarity'], sniffs[mouse]['trial_novelty'], sniffs[mouse]['trial_blank']]).T
        tr_cat.append(trm)
        
    ncat = tr_cat[mouse].shape[1] 
    tr_incl = copy.deepcopy(tr_cat) # copy category matrix to impose additional criteria
    
    # Now from the odor category matrix exclude some trials based on presentation no.
    for m in range(nses):
        for cat in range(ncat):
            if cat < 1: # for familiar odorants (1 first column of odor_cat_m), exclude preblock trials
                valid_trials = np.logical_and(sniffs[m]['trial_occur']>=fam_min, sniffs[m]['trial_occur']<=fam_max)
                valid_trials = np.logical_and(valid_trials, tr_incl[m][:,cat])
            else:
                valid_trials = np.logical_and(tr_incl[m][:,cat], sniffs[m]['trial_occur']>=nov_min)
                valid_trials = np.logical_and(tr_incl[m][:,cat], sniffs[m]['trial_occur']<=nov_max)
            
            tr_incl[m][:, cat] = valid_trials[:]*1
            
    return tr_cat, tr_incl


def av_by_occur(sniffs, trial_means, tr_cat):
    r"""
    Calculate average sniffing rate or pupil dilation by trial type occurence, together with SEM and N.
    Assumes you already have: 
        - 1 read-out for 1 trial (e.g. sniffing rate during selected bin)
        - a matrix (for each mouse/session) coding different odor categories, eg. output of 'select_trials_nov'

    Parameters
    ----------
    sniffs : output of 'import_sniff_mat' - list of dictionaries with all data about trials
    trial_means : a 2-dim array of size n trials x n mice/sessions
    tr_cat : a list of 2-dim arrays (output of select_trials_nov or similar), which allows to select trials by odor category

    Returns
    -------
    av_byoc 
    n_byoc 
    sem_byoc

    """
    
    max_occur = np.max([np.max(x['trial_occur']) for x in sniffs]) # what was the maximal number of 1 stimulus occurences in all data?
    nses = trial_means.shape[1] # no of sessions/mice
    ncat = tr_cat[0].shape[1] # number of odorant categories
    
    
    av_byoc = np.zeros([max_occur, nses, ncat])
    n_byoc = av_byoc.copy()
    sem_byoc = av_byoc.copy()
    
    for p in range(max_occur):
        for m in range(nses):
            for cat in range(ncat):
                which_tr = np.logical_and(tr_cat[m][:,cat], sniffs[m]['trial_occur'] == p+1)
                which_rows = sniffs[m]['trial_idx'][which_tr] - 1
                
                tmp_data = trial_means[which_rows, m]
                av_byoc[p, m, cat] = np.nanmean(tmp_data)
                n_byoc[p, m, cat] = tmp_data.size
                sem_byoc[p, m, cat] = np.nanstd(tmp_data) / np.sqrt(n_byoc[p, m, cat])

    return av_byoc, n_byoc, sem_byoc


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