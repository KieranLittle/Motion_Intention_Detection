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

from functions import read_file, resample, filter_smooth, segment_data, \
plot_segments, extract_features, MLR, filteremg, Support_Vector_Regression, plot_traj, predict_values, \
evaluate, plot_predicted_traj,  normalise_MVC

#%% READ TRAINING DATA AND TRAIN SVM

filename = 'patient_Kieran'
trial_num = '4'

dataset, amplitudes, emg_MVC = read_file(filename,trial_num) #compile into a single dataset (imu and emg)

#%%
# ## Resample Dataset

# dataset2 = resample(dataset)

# dataset_filtered = filter_smooth(dataset2)

# ## Segment data

# traj_segments,trajectory, peak_amplitude, peak_velocities,  mean_segment_velocity,time_3degrees,time_half,time_end, segment_line, segment_max_point = segment_data(dataset_filtered, amplitudes)

# plot_traj(trajectory,segment_line, segment_max_point)

# plot_segments(traj_segments, peak_amplitude, peak_velocities)

# extracted_features = extract_features(traj_segments,peak_amplitude, peak_velocities, mean_segment_velocity, time_3degrees,time_half,time_end)

# Training

# print('------------Support Vector Regression----------------\n')
# clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)

#%% READ AND EXTRACT FEATURES FROM TEST DATA

filename_list = ['patient_Kieran','patient_Adele','patient_Alessandro']
#test
i = 0
extracted_features_test = {}
trajectory_test = {}

for filename in filename_list:
    
    if filename == 'patient_Kieran':
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
        
        traj_segments_test,trajectory_test[i], peak_amplitude_test, peak_velocities_test,  mean_segment_velocity_test ,time_3degrees_test ,time_half_test,time_end_test, segment_line_test, segment_max_point_test = segment_data(dataset_filt_norm, amplitudes)
        
        #plot_segments(traj_segments_test, peak_amplitude, peak_velocities)
        #plot_traj(trajectory,segment_line, segment_max_point)
        
        extracted_features_test[i] = extract_features(traj_segments_test,peak_amplitude_test, peak_velocities_test, mean_segment_velocity_test, time_3degrees_test,time_half_test,time_end_test)
        i = i+1

full_combined_features = extracted_features_test[0].copy()
full_combined_trajectories = trajectory_test[0].copy()

for j in range(1,6):
  full_combined_features = full_combined_features.append(extracted_features_test[j], ignore_index=True, sort=False)

x = 149

for k in range(1,6):
    for t in range(0,len(trajectory_test[0])):
        
        #full_combined_trajectories[x] = trajectory_test[k][t]
        
        full_combined_trajectories[x] = trajectory_test[k][t]
        
        x=x+1
       
#%%
   

#clf,clf2,min_max_scaler, min_max_scaler2 
X_test,y_test = Support_Vector_Regression(full_combined_features)      

#full_combined_trajectories_test = full_combined_trajectories[full_combined_trajectories.keys()==X_test.index]
indices = X_test.index.tolist()

res = [full_combined_trajectories[i] for i in full_combined_trajectories.keys() if i in indices] 
#evaluate(full_combined_features)    
        
 #%% MAKE PREDICTIONS ON TEST DATA

#print('-----------Multiple Linear Regression---------------\n')
#X_train_scaled, X_test_scaled, y_train, y_test, df_predictions = MLR(extracted_features)
    
#print('------------Support Vector Regression----------------\n')
#clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)

#df_predictions_SVM = predict_values(extracted_features_test[1],clf,clf2,min_max_scaler, min_max_scaler2)
    
#evaluate(df_predictions_SVM)    


#%% PLOT PREDICTED TRAJECTORIES

plot_predicted_traj(X_test,res) #df_predictions_SVM
 


#%% EXAMPLE JERK TRAJECTORY

segment_num = 5

plt.figure(1,figsize=(15,8))
plt.subplot(4,1,1)
plt.plot(trajectory[segment_num]['time'],trajectory[segment_num]['imu'])

plt.subplot(4,1,2)
plt.plot(trajectory[segment_num]['time'][:-1],np.diff(trajectory[segment_num]['imu'],1))

plt.subplot(4,1,3)
plt.plot(trajectory[segment_num]['time'][:-2],np.diff(trajectory[segment_num]['imu'],2))

plt.subplot(4,1,4)
plt.plot(trajectory[segment_num]['time'][:-3],np.diff(trajectory[segment_num]['imu'],3))

plt.show()
jerk = np.diff(trajectory[segment_num]['imu'],3)

print(np.sum(jerk**2))









