
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
extracted_features = {}
df_trajectories = {}

# Define the segment cut-off point, this is based on the angular position
angle_cutoff = [1,2,5,10]

# define number of extracted points from segment
num_of_extracted_points = 10

# initialise index
i = 0

# Name to save final dictionary of datasets (one dataset per time window)
filename_save = 'dict.datasets_'+str(num_of_extracted_points)+'points'+'_change_filter3'

"""
    1. read_file(): reads in the raw data files 
    2. resample(): maybe this is not required anymore?
    3. filter_derivate(): filters all the signals and find the derivative signals
    4. normalise_MVC()
    5. split the trajectories into multiple segments per trajectory (based on number of angle_cutoffs given)
    6. extract features from each segment
: 
"""
for filename in filename_list:
    
    if filename == '3':
        trial_num_list = ['1','4']
    else:
        trial_num_list = ['1','2']
    
    for trial_num in trial_num_list:
        
        # compile different datafiles into a single dataset (imu, emg, stretch)
        dataset_combined_signals, amplitudes, emg_MVC = read_file(filename,trial_num)
        
        ## Resample Dataset
        dataset_RS = resample(dataset_combined_signals)
        
        # Filter and find the signal derivatives
        dataset_filtered = filter_derivate(dataset_RS)
        
        ## Normalise to MVC EMG
        dataset_norm = normalise_MVC(emg_MVC, dataset_filtered)
        
        ## Split the dataset into complete trajectories and then split into segments of different time windows
        traj_segments,df_trajectories[i], df_trajectories_info = segment_data(dataset_norm,angle_cutoff)
        
        # Extract features from segments
        extracted_features[i] = segment_extract_features(traj_segments,df_trajectories_info,filename,trial_num,num_of_extracted_points)
        
        i = i+1

#%%

"""
Combine all dataframes from trials into 1 dictionary containing 1 dataframe per window

"""
   
full_combined_features, full_combined_trajectories = combine_extracted_dataframes(extracted_features, df_trajectories, angle_cutoff)

        
#%%   
"""
Filter the features from the MJT R2 score

"""
sim_score = 0.85

filtered_mj_feat, filtered_mj_traj = filter_trajectories_by_MJT_sim(full_combined_trajectories,full_combined_features, angle_cutoff,sim_score)

#%% Detect outliers and drop rows

"""
Detect outliers and remove

"""

outlier_threshold= 0.2 # outliers must be found in 10% of the columns (of each row) or above, to be removed

filtered_mj_feat, trajectories,outlier_rows_per_dataframe = remove_outliers(filtered_mj_feat, filtered_mj_traj,outlier_threshold, angle_cutoff)
        

#%% Plot Trajectories
""" Plot trajectories
"""

# for traj in range(0,40):

#     plt.plot(filtered_mj_traj[traj]['time'], filtered_mj_traj[traj]['angular position'])
#     plt.show()
    
#%% Dataset Preparation
    
pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/data/03_processed/'+filename_save,"wb")
pickle.dump(filtered_mj_feat, pickle_out)
pickle_out.close()  
