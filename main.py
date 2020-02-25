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
from scipy import stats
import seaborn as sns   

from functions import read_file, resample, filter_smooth, segment_data, \
plot_segments, extract_features, MLR, filteremg, Support_Vector_Regression, plot_traj, predict_values, \
evaluate, plot_predicted_traj,  normalise_MVC, mjtg

#%% READ TRAINING DATA AND TRAIN SVM

filename = 'patient_Kieran'
trial_num = '4'

dataset, amplitudes, emg_MVC = read_file(filename,trial_num) #compile into a single dataset (imu and emg)

#%%
# ## Resample Dataset

dataset2 = resample(dataset)

dataset_filtered = filter_smooth(dataset2)

# ## Segment data

traj_segments,trajectory, df_trajectories = segment_data(dataset_filtered, amplitudes)

# plot_traj(trajectory,segment_line, segment_max_point)

plot_segments(traj_segments, df_trajectories['amplitude_change'], df_trajectories['peak_velocity'])

extracted_features = extract_features(traj_segments,df_trajectories)

# Training

# print('------------Support Vector Regression----------------\n')
# clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)

#%% READ AND EXTRACT FEATURES FROM TEST DATA

filename_list = ['patient_Kieran','patient_Adele','patient_Alessandro']

i = 0
extracted_features_test = {}
trajectory_test = {}
start_angle = {}

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
        
        traj_segments_test,trajectory_test[i], df_trajectories_test = segment_data(dataset_filt_norm, amplitudes)
        
        #plot_segments(traj_segments_test, peak_amplitude, peak_velocities)
        #plot_traj(trajectory,segment_line, segment_max_point)
        
        extracted_features_test[i] = extract_features(traj_segments_test,df_trajectories_test)
        i = i+1

#%% COMBINE FEATURES AND TRAJECTORY INTO 1 DICTIONARY EACH
        
        
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

#%% FILTER FEATURES BASED ON MJT R2 SCORE

from sklearn.metrics import r2_score

r2score = []

for p in range(0,len(full_combined_trajectories)):
    
    start = full_combined_trajectories[p].iloc[0,2]
    end = full_combined_trajectories[p].iloc[-1,2]
    time_start_ = full_combined_trajectories[p].iloc[0,0]
    time_end_ = full_combined_trajectories[p].iloc[-1,0]
    
    minimum_jerk_for_comp, vel = mjtg(start, end-start, 100, time_end_-time_start_)  #mjtg(start_angle[p], peak_amplitude[p]+start_angle[p], 101, time_end[p])
    
    #time = np.linspace(0,time_end[p]-0.01,int(time_end[p]*100))
    
    # plt.plot(time,minimum_jerk_for_comp[0:len(time)])
    # plt.plot(time,trajectory[p].iloc[0:len(time),2])
    
    plt.show()
    
    coefficient_of_dermination = r2_score(minimum_jerk_for_comp, full_combined_trajectories[p].iloc[0:len(minimum_jerk_for_comp),2])
    r2score.append(coefficient_of_dermination)
    
#print(r2score)

filtered_mj_traj = full_combined_trajectories.copy()
filtered_mj_feat = full_combined_features.copy()

for h in range(0, len(full_combined_trajectories.keys())):
    
    if r2score[h] < 0.9:
        del filtered_mj_traj[h]
        filtered_mj_feat.drop([h],inplace=True)
        
filtered_mj_feat = filtered_mj_feat.reset_index(drop = True)

filtered_mj_traj = {i: v for i, v in enumerate(filtered_mj_traj.values())}

        
#%%

sns.pairplot(filtered_mj_feat)
#filtered_mj_out_feat = filtered_mj_feat.copy()

#%% REMOVE OUTLIERS

filtered_mj_feat_out = filtered_mj_feat[(np.abs(stats.zscore(filtered_mj_feat)) < 3).all(axis=1)]

#### Remove from trajectories as well!

#%%
sns.pairplot(filtered_mj_feat_out)

#%%

for traj in range(0,20):

    plt.plot(filtered_mj_traj[traj]['time'], filtered_mj_traj[traj]['imu'])
    plt.show()

#%%   

#clf,clf2,min_max_scaler, min_max_scaler2 
X_test,y_test = Support_Vector_Regression(filtered_mj_feat_out)      

#full_combined_trajectories_test = full_combined_trajectories[full_combined_trajectories.keys()==X_test.index]
indices = X_test.index.tolist()

test_trajectories = [filtered_mj_traj[i] for i in filtered_mj_traj.keys() if i in indices] 
#evaluate(full_combined_features)    
        
 #%% MAKE PREDICTIONS ON TEST DATA

#print('-----------Multiple Linear Regression---------------\n')
#X_train_scaled, X_test_scaled, y_train, y_test, df_predictions = MLR(extracted_features)
    
#print('------------Support Vector Regression----------------\n')
#clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)

#df_predictions_SVM = predict_values(extracted_features_test[1],clf,clf2,min_max_scaler, min_max_scaler2)
    
#evaluate(df_predictions_SVM)    


#%% PLOT PREDICTED TRAJECTORIES

plot_predicted_traj(X_test,test_trajectories) #df_predictions_SVM
 


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









