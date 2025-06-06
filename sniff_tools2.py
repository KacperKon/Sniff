# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:36:45 2021
@author: kkondrakiewicz

Functions for analyzing breathing rate and pupil diameter from head-fixed animals.
The current version, v2, is using lists of 2-dim arrays instead of 3-dim arrays,
which allows to analyze datasets with animals reveiving unequal number of trials.
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
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM 
from statsmodels.stats.multitest import fdrcorrection as fdr
from scipy import stats
from matplotlib.patches import Rectangle


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

def import_sniff_mat_select(paths_list):
    r"""
    Import pre-processed behavioral data on sniffing and trial structure.
    This version requires specyfing from which folders exactly you want to import.
    The assumption is that there is a folder per session organization.

    Parameters
    ----------
    paths_list : list of paths, e.g. paths_list[0] = 'C:\\data\\220308_KK012'

    Returns
    -------
    sniffs : a list of dictionaries 
    Should contain all the fields from imported Matlab structures (in a single dictionary for each mouse / session).
    
    """
    nses = len(paths_list) # number of all recording sessions
    
    sniffs = []
    
    for mouse in range(nses):
        tmp_dict = {}
        tmp_files = glob.glob(paths_list[mouse] + '/*.mat')
        for file in tmp_files:
            tmp_dict = {**tmp_dict, **io.loadmat(file, squeeze_me=True)}
            
        sniffs.append(tmp_dict)
        
    return sniffs


def bin_sniff(sniffs, nframes, bsln_start, odor_start, the_bin, binsize, classic_tracker = False):
    r"""
    Based on imported .mat strutcures, calculate sniffing rate in bins.

    Parameters
    ----------
    sniffs : output of the import_sniff_mat function
    nframes : how many frames you want in each trial from the sniff camera; everything after will be ignored
    bsln_start : start of baseline for breathing rate normalization [sec]
    odor_start : [sec] - it's also end of baseline period
    the_bin : list of 2 elements [sec] - edges of your bin of interest - size of this bin might be different from the binsize
    binsize : [sec] - size of bin for calculation of breatinh rate
    classic_tracker : set to True if you want to use old FLIR tracker algorithm results and not DeepSniff

    Returns
    -------
    sniff_ons : list (n mice/sessions) of 2-dim arrays (n trials x n frames) with binarized sniff onsets (onset == 1) 
    sniff_list : the same data as above, but stored as list of lists of timestamps (mice >> trials) - handy for rasterplots
    sniff_hist : list (n mice/sessions) of 2-dim arrays (n trials x n bins)
                a histogram of breaths count for each bin, normalized by the bin size (so in Hz) 
    sniff_delhist : as above, but the breathing rate from the baseline period is subtracted
    sniff_mybin : list (n mice) of 1-dim arrays (n trials) - sniffing rate for one selected bin, 
                which can be of different size than the 'binsize', subtracted from the baseline

    """

    # Extract basic variables 
    nses = len(sniffs)
    ntrials = [x['trial_idx'].size for x in sniffs]
    sr = [x['samp_freq'] for x in sniffs]
    if np.std(sr) != 0:
        print('Different sessions were acquired with different camera sampling rates! Not supported :(')
    else:
        sr = sr[0]
    bin_edges = np.arange(0, nframes, binsize*sr)
    bsln_size = (int(odor_start*sr) - int(bsln_start*sr)) / sr
    
    # Prepare structures to fill with the data
    sniff_ons = [np.zeros([nt, nframes]) for nt in ntrials] # list (n mice) of arrays (n trials x n mice)   
    sniff_hist    = [np.zeros([nt, bin_edges.size-1]) for nt in ntrials]
    sniff_delhist = [np.zeros([nt, bin_edges.size-1]) for nt in ntrials]
    sniff_mybin = [np.zeros([nt]) for nt in ntrials]
    sniff_list = []

    # Calculate sniffing rates
    for m in range(nses):
        tmp_list = []
        for tr in range(ntrials[m]):
            if classic_tracker == False:
                in_win = sniffs[m]['ml_inh_onsets'][tr] <= nframes-1 # select inhalations no later than 'nframes'
                sniff_ons[m][tr, sniffs[m]['ml_inh_onsets'][tr][in_win]] = 1 # code sniff onsets as 1
                tmp_list.append(sniffs[m]['ml_inh_onsets'][tr][in_win]) # store inhalation onsets also in a list (for raster plots)
            elif classic_tracker == True:
                in_win = sniffs[m]['ft_inh_onsets'][tr] <= nframes-1
                sniff_ons[m][tr, sniffs[m]['ft_inh_onsets'][tr][in_win]] = 1 
                tmp_list.append(sniffs[m]['ft_inh_onsets'][tr][in_win])
              
            sniff_hist[m][tr,:] = np.histogram(tmp_list[tr], bin_edges)[0]/binsize
            bsln = np.sum(sniff_ons[m][tr, int(bsln_start*sr) : int(odor_start*sr) ])/bsln_size
            sniff_delhist[m][tr,:] = sniff_hist[m][tr,:] - bsln
                
            sniff_mybin[m][tr] = np.sum(sniff_ons[m][tr, int(the_bin[0]*sr) : int(the_bin[1]*sr)])/(the_bin[1] - the_bin[0]) - bsln
                
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
    sniff_gauss : list (n mice/sessions) of 2-dim arrays (n trials x n frames) with instantenous sniffing rate
    sniff_delta : as above, but with baseline sniffing subtracted

    """
    
    nses = len(sniff_ons)
    ntrials = [x.shape[0] for x in sniff_ons]
    nframes = sniff_ons[0].shape[1]
    
    sniff_gauss = [np.zeros([nt, nframes]) for nt in ntrials]
    sniff_delta = [np.zeros([nt, nframes]) for nt in ntrials]
    
    
    for m in range(nses):
        for tr in range(ntrials[m]):
            sniff_gauss[m][tr, :] =  gaussian_filter1d(sniff_ons[m][tr,:], sigma*sr, mode = 'reflect') *sr
            bsl = np.mean(sniff_gauss[m][tr, int(bsln_start*sr):int(odor_start*sr)])
            sniff_delta[m][tr, :] = sniff_gauss[m][tr, :] - bsl
     
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
                valid_trials = np.logical_and(sniffs[m]['trial_occur']>=nov_min, sniffs[m]['trial_occur']<=nov_max)
                valid_trials = np.logical_and(valid_trials, tr_incl[m][:,cat])

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
    av_byoc : 3-dim array (n occurences x n mice/sessions x n odor categories) - average
        - If some stimuli occur less time then max_occur, respective entry will be NaN. 
    n_byoc : as above; N of samples used for the calculation
    sem_byoc : as above; standard error of the mean

    """
    
    max_occur = np.max([np.max(x['trial_occur']) for x in sniffs]) # what was the maximal number of 1 stimulus occurences in all data?
    nses = len(sniffs)
    ncat = tr_cat[0].shape[1] # number of odorant categories; we assume the same for all mice
    
    
    av_byoc = np.zeros([max_occur, nses, ncat])
    n_byoc = np.zeros([max_occur, nses, ncat])
    sem_byoc = np.zeros([max_occur, nses, ncat])
    
    for p in range(max_occur):
        for m in range(nses):
            for cat in range(ncat):
                which_tr = np.logical_and(tr_cat[m][:,cat], sniffs[m]['trial_occur'] == p+1)
                which_rows = sniffs[m]['trial_idx'][which_tr] - 1
                
                tmp_data = trial_means[m][which_rows]
                av_byoc[p, m, cat] = np.nanmean(tmp_data)
                n_byoc[p, m, cat] = tmp_data.size
                sem_byoc[p, m, cat] = np.nanstd(tmp_data) / np.sqrt(n_byoc[p, m, cat])

    return av_byoc, n_byoc, sem_byoc


def gen_ast(pval):
    """
    Generate aterisks to illustrate p value on the plots.
    """
    
    if pval >= 0.05:
        ast = ' ns'
    elif (pval < 0.05) & (pval >= 0.01):
        ast = '*'
    elif (pval < 0.01) & (pval >= 0.001):
        ast = '**'
    elif pval < 0.001:
        ast = '***'
    
    return ast

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
        
        if len(camlog_files) > 0:
            log,comm = parse_cam_log(camlog_files[m])
            pup_ts.append(np.array(log['timestamp']))
        else:
            pup_ts = np.nan
            print("Could not find .camlog file - frames have no timestamps")
        
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
                tmp_data = np.interp(np.arange(len(tmp_data)), np.arange(len(tmp_data))[np.isnan(tmp_data) == False], tmp_data[np.isnan(tmp_data) == False])
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


#%% Functions written by Michiel Camps to process video data (pupil, motion etc.)

# Some boilerplate functions for loading keypoints and timestamps etc.
def load_keypoints(wd):
    keypointfiles = sorted(wd.glob("*.keypoints.npy"))
    alldata = [np.load(k, allow_pickle=True) for k in keypointfiles]
    if len(alldata) == 0:
        print("[-] No keyopint files found")
    else:
        print(f"[+] {len(alldata)} keypoint files loaded")
    return alldata

def load_ENERGY(wd):
    return np.concatenate([np.load(f) for f in sorted(wd.glob("*_energy.npy"))])

def load_avis(wd):
    return sorted(wd.glob("*.avi"))

def timestamp_from_camlog(camlog):
    data = np.loadtxt(camlog, delimiter=",")
    return data[:,1] / 1e9  # in seconds, I think

def load_timestamps(wd):
    camlog = wd.glob("*.camlog")
    try: 
        camlog = next(camlog)
        print("[+] Camlog found, loading timestamps")
    except StopIteration:
        print("[-] No camlog found, no timestamps available")
        return []
    return timestamp_from_camlog(camlog)

def _extract_cat(keypoints, key):
    return np.concatenate([p.item()[key] for p in keypoints])

def pupilsize(keypoints):
    # multiply by π / 4 if you want to get approximate size in pixels
    return _extract_cat(keypoints, "pupil_size")
    
def eyesize(keypoints):
    # multiply by π / 4 if you want to get approximate size in pixels
    return _extract_cat(keypoints, "eye_size")

def pupilpos(keypoints):
    return _extract_cat(keypoints, "pupil_pos")

def eyepos(keypoints):
    return _extract_cat(keypoints, "eye_pos")

def certainty(keypoints):
    return _extract_cat(keypoints, "certainty")

def split_trials(ts, series):
    splits = np.where(np.diff(ts) > 1)[0] + 1  # ITI > 1s required
    splits = np.append(splits, min(len(series), len(ts)))
    splits = np.insert(splits, 0, 0)
    def stacker(s):
        return [np.take(s, np.asarray(range(splits[i], splits[i+1])), 0) for i in range(len(splits)-1)]
    return stacker(ts), stacker(series)


#%% Functions for standard plots and statistics (for the paper)

def rm_anova(data, pairs):
    r"""
    Run one-way repeated-measures ANOVA on your data, 
    followed by paired t-tests with FDR correction.

    Parameters
    ----------
    data : array of shape subjects x repeated measures
    pairs : array of shape 2 x comparisons:
        indexes of columns to compare with the t-test

    Returns
    -------
    anova_res, pvals, pvals_adj

    """
    
    nmice = data.shape[0]
    ngroups = data.shape[1]

    grav = np.reshape(data, [np.size(data),],'F')
    group = np.arange(0, ngroups)
    group = np.repeat(group, nmice)
    
    df = pd.DataFrame({'Dependent': grav, 'Group': group, 'Mouse': np.tile(np.arange(nmice), ngroups)})
    excl_mice = df['Mouse'][np.isnan(df['Dependent'])].values
    
    anova_res = AnovaRM(data=df[~df['Mouse'].isin(excl_mice)],
                depvar='Dependent', subject='Mouse', within=['Group']).fit() 
    print(anova_res) 
    
    pvals = []
    for ii in range(pairs.shape[1]):    
        stat, p = stats.ttest_rel(grav[group==pairs[0, ii]], grav[group==pairs[1, ii]],
                                  nan_policy = 'omit')
        pvals.append(p)
    pvals = np.array(pvals)
    
    pvals_adj = fdr(pvals)[1]
    
    print('       P values (uncorrected):')
    print(np.round(pvals, 3))
    
    print('           P values (FDR):')
    print(np.round(pvals_adj, 3))
    
    return anova_res, pvals, pvals_adj


def plot_bars(data, ax=[], colors=[], mfc=[], labels=[], max_jit = 0, ms=30):
    
    r"""
    Plot barplot + connected points to summarize within-subject differences 
    between conditions.
    
    data is an array subjects x conditions 

    """
    
    nmice = data.shape[0]
    ngroups = data.shape[1]
    xjit = (np.random.rand(nmice, ngroups) - 0.5) *max_jit

    grav = np.reshape(data, [np.size(data),],'F')
    group = np.arange(0, ngroups)
    group = np.repeat(group, nmice)
    group = group + np.reshape(xjit, [np.size(xjit),], 'F')
    
    if mfc==[]:
        mfc=['gray']*ngroups
    if ax==[]:
        fig, ax = plt.subplots(figsize = (1.8, 2.7), dpi = 300)
        
    ind_color = np.repeat(colors, nmice)
    ind_mfc = np.repeat(mfc, nmice)
        
    tmp_av = np.nanmean(data, 0)
    tmp_se = np.nanstd(data, 0) / np.sqrt(nmice)

    for m in range(nmice):
        line_to_plot = data[m,:]
        x_to_plot = np.arange(0, ngroups) + xjit[m,:]
        nan_idx = np.where(np.isnan(line_to_plot) == True)[0]
        if len(nan_idx)==1 and nan_idx[0] < max(group):
            line_to_plot[nan_idx] = np.mean([line_to_plot[nan_idx-1], line_to_plot[nan_idx+1]])
        ax.plot(x_to_plot, line_to_plot, color = 'gray', zorder=0, linewidth = 0.6)
        
    ax.scatter(group, grav, marker = 'o', color = ind_color, facecolors = ind_mfc, s = ms, zorder=1, alpha = 0.5)
    ax.bar(np.arange(ngroups), tmp_av, yerr = tmp_se, fill=False, edgecolor = 'k', lw = 1)
        
      
    if labels != []:
        ax.set_xticks(np.arange(ngroups))
        ax.set_xticklabels(labels, rotation = 0, fontsize=8)

    plt.tight_layout()

    return ax


def add_significance(ax, pairs, pvals, thr=0.05, line_dist=0.05, ast_dist=0.05):
    r"""
    Add horizontal bars between significantly different conditions
    to an existing plot where different groups are plotted on x axis
    """
    is_sig = pvals < thr
    
    line_h = ax.get_ylim()[1] - 0.05*ax.get_ylim()[1] # how high to plot annotation lines
    for ii in range(pairs.shape[1]):
        if is_sig[ii]:
            ax.plot([pairs[0,ii], pairs[1,ii]], [line_h, line_h], lw=1, color = 'k')            
            sig_label = gen_ast(pvals[ii])
            ax.annotate(sig_label, [np.mean([pairs[0,ii], pairs[1,ii]]), line_h+ast_dist], 
                        size = 15, ha = 'center')
    
            line_h = line_h - line_dist
            
    return ax


def plot_tseries(data, tvec = [], ax=[], error=[], colors=[], line_styles=[],
                 line_alpha=1, error_alpha = 0.2, lw=1, labels=[]):
    
    r"""
    Create a plot with multiple time series (e.g., avergae traces) plotted on
    top of each other, optionally with errorbar.
    Input data: time points in rows, different series in columns.
    """
    
    nseries = data.shape[1]
    npoints = data.shape[0]
    
    if ax==[]:
        fig, ax = plt.subplots(figsize = (3, 2), dpi = 600)
    if tvec==[]:
        tvec = np.arange(npoints)
    if colors==[]:
        colors = ['gray']*nseries
    if line_styles==[]:
        line_styles = ['-']*nseries
    
    for s in range(nseries):    
        ax.plot(tvec, data[:,s], c=colors[s], ls=line_styles[s], lw=lw,
                alpha=line_alpha)
        
        if len(error)>0:
            ax.fill_between(tvec, data[:,s]+error[:,s], data[:,s]-error[:,s], 
            color=colors[s], alpha=error_alpha, lw=0.01)
            
    if labels!=[]:
        ax.legend(labels = labels, loc='upperright')
            
    plt.tight_layout()

    return ax


def add_opto(ax):
    r"""
    Add patches ilustrating the opto stimulation
    """

    # For blue light
    cmax = 150
    ax.add_patch(Rectangle((0, -0.4), 3, 0.15, color = plt.cm.Blues(cmax)))
    ax.imshow([[cmax, 10], [cmax, 10]], cmap = plt.cm.Blues, interpolation = 'bicubic',\
               extent=[3, 4, -0.4, -0.235], vmin = 0, vmax = 255, aspect='auto')
    # and for red   
    ax.add_patch(Rectangle((0.7, -0.218), 0.5, 0.15, color = 'tomato')) 

    return ax


def plot_curves(data, error=[], ax=[], maxpoints=[], colors=[], line_styles=[], 
                mfc=[], labels=[], jit=0, ms=4):
    r"""
    PLot a series of lines with optional error bars.
    Useful for plotting habituation curves.
    Input data: array of size time points x n series (groups, conditions etc.)
    """
    
    npoints = data.shape[0]
    nseries = data.shape[1]
    jits = np.linspace(-jit, jit, nseries)
    
    if maxpoints==[]:    
        maxpoints=npoints
    tvec = np.arange(1, maxpoints+1)

    if colors==[]:
        colors = ['gray']*nseries
    if line_styles==[]:
        line_styles = ['-']*nseries
    if mfc==[]:
        mfc=['gray']*nseries
    if len(error)==0:
        error==np.zeros([npoints, nseries])
    if labels==[]:
        labels=[[] for i in range(nseries)]
        
    if ax==[]:
        fig, ax = plt.subplots(figsize = (3, 2), dpi = 600)
        
    for s in range(nseries):    
        ax.errorbar(tvec+jits[s], data[:maxpoints,s], error[:maxpoints,s], 
        color=colors[s], ls=line_styles[s], lw=1, fmt = 'o', ms=ms, mfc=mfc[s],
        label = labels[s])
        
        
    plt.tight_layout()
        
    return ax
