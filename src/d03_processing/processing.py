
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



#%% READ TRAINING DATA AND TRAIN SVM

"""
Single dataset for testing
"""

sub_num = '3'
trial_num = '1'

dataset, amplitudes, emg_MVC = read_file(sub_num,trial_num) #compile into a single dataset (imu and emg)

dataset2 = resample(dataset)
dataset_filtered = filter_smooth(dataset2)

traj_segments,trajectory, df_trajectories = segment_data(dataset_filtered, amplitudes)

plot_traj(trajectory,df_trajectories['segment_cut_off'], df_trajectories['segment_max_point'])

plot_segments(traj_segments, df_trajectories['amplitude_change'], df_trajectories['peak_velocity'])

extracted_features = extract_features(traj_segments,df_trajectories)

# Training

# print('------------Support Vector Regression----------------\n')
# clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)
X_test,y_test = Support_Vector_Regression(extracted_features)

#%% READ AND EXTRACT FEATURES FROM TEST DATA

"""
READ AND EXTRACT FEATURES FROM TEST DATA

"""

filename_list = ['1','2','3']
    
i = 0
extracted_features_test = {}
trajectory_test = {}
start_angle = {}

for filename in filename_list:
    
    if filename == '3':
        trial_num_list = ['1','4']
    else:
        trial_num_list = ['1','2']
    
    for trial_num in trial_num_list:
    
        dataset_test, amplitudes, emg_MVC = read_file(filename,trial_num) #compile into a single dataset (imu and emg)
        
        ## Resample Dataset
        
        dataset_test_RS = resample(dataset_test)
        
        dataset_test_filtered = filter_smooth(dataset_test_RS)
        
        ## Normalise to MVC EMG
        dataset_filt_norm = normalise_MVC(emg_MVC, dataset_test_filtered)
        
        ## Segment data
        
        traj_segments_test,trajectory_test[i], df_trajectories_test = segment_data(dataset_filt_norm, amplitudes)
        
        #plot_segments(traj_segments_test, peak_amplitude, peak_velocities)
        #plot_traj(trajectory,segment_line, segment_max_point)
        
        extracted_features_test[i] = extract_features(traj_segments_test,df_trajectories_test)
        i = i+1

#%%
"""
Combine all the features and trajectories into 1 dictionary each

"""
        
full_combined_features = extracted_features_test[0].copy()
full_combined_trajectories = trajectory_test[0].copy()

for j in range(1,6):
  full_combined_features = full_combined_features.append(extracted_features_test[j], ignore_index=True, sort=False)

x = 149

col = []

for i in range(1,7):
    col.append('t{}'.format(i))
    col.append('s{}'.format(i))
    col.append('bb{}'.format(i))
    col.append('tb{}'.format(i))
    col.append('ad{}'.format(i))
    col.append('pm{}'.format(i))
    col.append('theta{}'.format(i))
    col.append('theta_d{}'.format(i))
    col.append('theta_dd{}'.format(i))
    col.append('s_d{}'.format(i))
    col.append('s_dd{}'.format(i))
    col.append('bb_d{}'.format(i))
    col.append('tb_d{}'.format(i))
    col.append('ad_d{}'.format(i))
    col.append('pm_d{}'.format(i))
    col.append('bb_dd{}'.format(i))
    col.append('tb_dd{}'.format(i))
    col.append('ad_dd{}'.format(i))
    col.append('pm_dd{}'.format(i))

# col.extend(['stretch_mean','bb_mean','tb_mean','ad_mean','pm_mean','stretch_var','bb_var','tb_var','ad_var','pm_var'])
col.extend(['peak_velocity','peak_amplitude','mean_velocity','time_end','time_half'])
            
full_combined_features.columns = col

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
    
    if r2score[h] < 0.8:
        del filtered_mj_traj[h]
        filtered_mj_feat.drop([h],inplace=True)
        
filtered_mj_feat = filtered_mj_feat.reset_index(drop = True)
filtered_mj_traj = {i: v for i, v in enumerate(filtered_mj_traj.values())}

#%% Detect outliers and drop rows

lst = []
cols = [i for i in range(0,len(filtered_mj_feat.columns)+1)]
Outliers_to_drop = detect_outliers(filtered_mj_feat,20,list(filtered_mj_feat.columns.values))

filtered_mj_feat_out = filtered_mj_feat.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

"DROP TRAJECTORIES"
filtered_mj_traj_out = {i: v for i, v in enumerate(filtered_mj_traj.values())}


#%% Plot Trajectories
""" Plot trajectories
"""

for traj in range(0,40):

    plt.plot(filtered_mj_traj[traj]['time'], filtered_mj_traj[traj]['angular position'])
    plt.show()
    
#%% Dataset Preparation

dataset_processed = filtered_mj_feat_out

dataset_processed.to_csv(r'data/03_processed/data_processed',index=False,header=True)
