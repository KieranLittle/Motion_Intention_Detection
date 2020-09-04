
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:37 2020

@author: Kieran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import pickle
import seaborn as sns   

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from src.d00_utils.functions import *
from scipy import stats

import sys
import os

#%% READ AND EXTRACT FEATURES FROM TEST DATA

"""
READ AND EXTRACT FEATURES FROM TEST DATA

"""

# list of the subject numbers in the data files
filename_list = ['1','2','3']

# initialise dictionaries
extracted_features_test = {}
df_selected_features= {}
trajectory_test = {}
start_angle = {}

# initialise index
i = 0

"""
    1. read_file(): reads in the raw data files 
    2. resample():

"""
for filename in filename_list:
    
    if filename == '3':
        trial_num_list = ['1','4']
    else:
        trial_num_list = ['1','2']
    
    for trial_num in trial_num_list:
        
        # compile different datafiles into a single dataset (imu, emg, stretch)
        dataset_test, amplitudes, emg_MVC = read_file(filename,trial_num)
        
        ## Resample Dataset
        dataset_test_RS = resample(dataset_test)
        
        dataset_test_filtered = filter_derivate(dataset_test_RS)
        
        ## Normalise to MVC EMG
        dataset_filt_norm = normalise_MVC(emg_MVC, dataset_test_filtered)
        
        ## Segment data
        
        traj_segments_test,trajectory_test[i], df_trajectories_test = segment_data(dataset_filt_norm, amplitudes)
        
        #plot_segments(traj_segments_test, peak_amplitude, peak_velocities)
        #plot_traj(trajectory,segment_line, segment_max_point)
        
        extracted_features_test[i], df_selected_features[i],col = extract_features(traj_segments_test,df_trajectories_test,filename,trial_num)
        i = i+1

#%%
"""
Combine all the features and trajectories into 1 dictionary each

"""

full_combined_features = {}

for window in range(0,4):
    full_combined_features[window] = extracted_features_test[0][window].copy()

    for j in range(1,6):
        
        full_combined_features[window] = full_combined_features[window].append(extracted_features_test[j][window], ignore_index=True, sort=False)


        
full_combined_trajectories = trajectory_test[0].copy()
x=0

for k in range(1,6):
    for t in range(0,len(trajectory_test[0])):
        
        full_combined_trajectories[x] = trajectory_test[k][t]
        x=x+1
   
        
#%%   
"""
Filter the features from the MJT R2 score

"""

from sklearn.metrics import r2_score

r2score = []

"Create the minimum jerk trajectory for comparison with traj"

for p in range(0,len(full_combined_trajectories)):
    
    theta_col = full_combined_trajectories[p].columns.get_loc("angular position")
    theta_d_col = full_combined_trajectories[p].columns.get_loc("angular position")
    
    start = full_combined_trajectories[p].iloc[0,theta_col]
    end = full_combined_trajectories[p].iloc[-1,theta_col]
    time_start_ = full_combined_trajectories[p].iloc[0,0]
    time_end_ = full_combined_trajectories[p].iloc[-1,0]
    
    minimum_jerk_for_comp, vel = mjtg(start, end-start, 100, time_end_-time_start_)
    time = np.linspace(time_start_,time_end_-0.01,len(minimum_jerk_for_comp))
    
    coefficient_of_dermination = r2_score(minimum_jerk_for_comp, full_combined_trajectories[p].iloc[0:len(minimum_jerk_for_comp),theta_col])
    r2score.append(coefficient_of_dermination)

"""
Filter features based on R2 score
"""

filtered_mj_traj = full_combined_trajectories.copy()
filtered_mj_feat = full_combined_features.copy()

for h in range(0, len(full_combined_trajectories.keys())):
    
    if r2score[h] < 0.85:
        del filtered_mj_traj[h]
        
        for i in range(0,4): ###NUMBER OF DATASETS
        
            #dataset_slct = filtered_mj_feat[i]
            filtered_mj_feat[i].drop([h],inplace=True)

for i in range(0,4):
    filtered_mj_feat[i] = filtered_mj_feat[i].reset_index(drop=True)
            
        
            #filtered_mj_feat = filtered_mj_feat.reset_index(drop = True)

# RESET INDEX:
filtered_mj_traj = {i: v for i, v in enumerate(filtered_mj_traj.values())}

#%% Detect outliers and drop rows

# lst = []
# cols = [i for i in range(0,len(filtered_mj_feat.columns)+1)]


for i in range(0,4):
    Outliers_to_drop = detect_outliers(filtered_mj_feat[i],25,list(filtered_mj_feat[i].columns.values))
    print(len(Outliers_to_drop))
    filtered_mj_feat[i].drop(Outliers_to_drop, axis = 0,inplace=True)
    filtered_mj_feat[i]=filtered_mj_feat[i].reset_index(drop=True)

    #filtered_mj_feat_out[i] = 
    
# "DROP TRAJECTORIES using trajectories, not completed"
# filtered_mj_traj_out = {i: v for i, v in enumerate(filtered_mj_traj.values())}


#%% Plot Trajectories
""" Plot trajectories
"""

for traj in range(0,40):

    plt.plot(filtered_mj_traj[traj]['time'], filtered_mj_traj[traj]['angular position'])
    plt.show()
    
#%% Dataset Preparation
    
# dataset_processed = filtered_mj_feat

# dataset_processed.to_csv(r'data/03_processed/data_processed_dictionary',index=False,header=True)

pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/data/03_processed/dict.datasets',"wb")
pickle.dump(filtered_mj_feat, pickle_out)
pickle_out.close()  
