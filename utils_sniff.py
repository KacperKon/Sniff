# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:36:45 2021

@author: kkondrakiewicz
"""

import re
import pandas as pd

# Copied from GitBucket of Jao (creator of mptracker)
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

# Copied from GitBucket of Jao (creator of mptracker)
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