#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:30:01 2020

@author: Kieran
"""
### Imports

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
import pickle

#%% Read data

file = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/data/03_processed/dict.datasets','rb')

# dump information to that file
dataset_dict = pickle.load(file)

# close the file
file.close()

#%% Test Modelling using a single dataset (This is just for testing, not for producing final results)

X = dataset_dict[0].drop(['Peak Amplitude','Peak Velocity','Mean Velocity','T_end', 'T_half'],axis=1)
    
y = dataset_dict[0][['Peak Amplitude','Peak Velocity','Mean Velocity', 'T_half']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train_dur = y_train['T_half']
y_train_mv = y_train['Mean Velocity']
y_train_pa = y_train['Peak Amplitude']
        
y_test_dur = y_test['T_half']
y_test_mv = y_test['Mean Velocity']
y_test_pa = y_test['Peak Amplitude']
    
#mean_peak_velocity = y.peak_velocity.mean()
mean_time_half = y.T_half.mean()
mean_peak_amplitude = y['Peak Amplitude'].mean()
mean_mean_velocity = y['Mean Velocity'].mean()

### Scaling

# Scale the train data to range [0 1] and scale the test data according to the train data
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_scaled = min_max_scaler.fit_transform(X_train)
# X_test_scaled = min_max_scaler.transform(X_test)

### Baseline regression models

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# cv_res_df_mv = baseline_regression_models(X_train, y_train_mv, mean_mean_velocity, kfolds)
# cv_res_df_ht =  baseline_regression_models(X_train, y_train_dur, mean_time_half, kfolds)
# cv_res_df_pa =  baseline_regression_models(X_train, y_train_pa, mean_peak_amplitude, kfolds)


#%% 

"""
Create a list for each sensor containing their features 

"""

predictors=list(X_train)

imu_col = []
stretch_col = []
emg_col = []

for i in range(1,7):
    imu_col.append('time{}'.format(i))
    imu_col.append('pos{}'.format(i))
    imu_col.append('vel{}'.format(i))
    imu_col.append('acc{}'.format(i))
    
    stretch_col.append('time{}'.format(i))
    stretch_col.append('stretch{}'.format(i))
    
    emg_col.append('time{}'.format(i))
    emg_col.append('bb{}'.format(i))
    emg_col.append('tb{}'.format(i))
    emg_col.append('ad{}'.format(i))
    emg_col.append('pm{}'.format(i))

#%% Make a loop to test all datasets

"""
This cell returns 


"""

cv_means = []
cv_std = []
cv_results = {}

cv_res_df_mv_comb = {}
cv_res_df_ht_comb = {}
cv_res_df_pa_comb = {}

feat_imp_mv_comb = {}
feat_imp_ht_comb = {}
feat_imp_pa_comb = {}

sensor_imp_mv_comb = {}
sensor_imp_ht_comb = {}
sensor_imp_pa_comb = {}

cv_res_df_mv_imu = {}
feat_imp_mv_imu = {}

cv_res_df_ht_imu = {}
feat_imp_ht_imu = {}
 
cv_res_df_pa_imu = {}
feat_imp_pa_imu = {}

cv_res_df_mv_stretch = {}
feat_imp_mv_stretch = {}

cv_res_df_ht_stretch = {}
feat_imp_ht_stretch = {}

cv_res_df_pa_stretch = {}
feat_imp_pa_stretch = {}

cv_res_df_mv_emg= {}
feat_imp_mv_emg = {}

cv_res_df_ht_emg = {}
feat_imp_ht_emg= {} 

cv_res_df_pa_emg = {}
feat_imp_pa_emg = {}

i = 0

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
n_jobs=2

for dataset_slt in dataset_dict.values():    
    
    X = dataset_slt.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','T_end', 'T_half'],axis=1)
    y = dataset_slt[['Peak Amplitude','Peak Velocity','Mean Velocity', 'T_half']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    y_train_dur = y_train['T_half']
    y_train_mv = y_train['Mean Velocity']
    y_train_pa = y_train['Peak Amplitude']
        
    y_test_dur = y_test['T_half']
    y_test_mv = y_test['Mean Velocity']
    y_test_pa = y_test['Peak Amplitude']
    
    #mean_peak_velocity = y.peak_velocity.mean()
    mean_time_half = y.T_half.mean()
    mean_peak_amplitude = y['Peak Amplitude'].mean()
    mean_mean_velocity = y['Mean Velocity'].mean()
    
    X_train_imu = X_train[imu_col]
    X_test_imu = X_test[imu_col]
    
    X_train_stretch = X_train[stretch_col]
    X_test_stretch = X_test[stretch_col]
    
    X_train_emg = X_train[emg_col]
    X_test_emg = X_test[emg_col]
    
    ## ALL SIGNALS
    cv_res_df_mv_comb[i],feat_imp_mv_comb[i], sensor_imp_mv_comb[i] =  baseline_regression_models(X_train, y_train_mv, mean_mean_velocity, kfolds, combined=1)
    cv_res_df_ht_comb[i],feat_imp_ht_comb[i], sensor_imp_ht_comb[i] =  baseline_regression_models(X_train, y_train_dur, mean_time_half, kfolds, combined=1)
    cv_res_df_pa_comb[i],feat_imp_pa_comb[i], sensor_imp_pa_comb[i] =  baseline_regression_models(X_train, y_train_pa, mean_peak_amplitude, kfolds, combined=1)
    
    # ## IMU
    cv_res_df_mv_imu[i],feat_imp_mv_imu[i]  =  baseline_regression_models(X_train_imu, y_train_mv, mean_mean_velocity, kfolds)
    cv_res_df_ht_imu[i],feat_imp_ht_imu[i] =  baseline_regression_models(X_train_imu, y_train_dur, mean_time_half, kfolds)
    cv_res_df_pa_imu[i],feat_imp_pa_imu[i] =  baseline_regression_models(X_train_imu, y_train_pa, mean_peak_amplitude, kfolds)
    
    # ## Stretch 
    cv_res_df_mv_stretch[i], feat_imp_mv_stretch[i] =  baseline_regression_models(X_train_stretch, y_train_mv, mean_mean_velocity, kfolds)
    cv_res_df_ht_stretch[i], feat_imp_ht_stretch[i] =  baseline_regression_models(X_train_stretch, y_train_dur, mean_time_half, kfolds)
    cv_res_df_pa_stretch[i], feat_imp_pa_stretch[i] =  baseline_regression_models(X_train_stretch, y_train_pa, mean_peak_amplitude, kfolds)
    
    # # EMG
    cv_res_df_mv_emg[i], feat_imp_mv_emg[i] =  baseline_regression_models(X_train_emg, y_train_mv, mean_mean_velocity, kfolds)
    cv_res_df_ht_emg[i], feat_imp_ht_emg[i] =  baseline_regression_models(X_train_emg, y_train_dur, mean_time_half, kfolds)
    cv_res_df_pa_emg[i], feat_imp_pa_emg[i] =  baseline_regression_models(X_train_emg, y_train_pa, mean_peak_amplitude, kfolds)
    
    
    i+=1

# Save all the results to a dictionary based on the sensor

comb_results = {}
comb_results['cv_res_mv'] = cv_res_df_mv_comb
comb_results['feat_imp_mv'] = feat_imp_mv_comb
comb_results['sensor_imp_mv'] = sensor_imp_mv_comb

comb_results['cv_res_ht'] = cv_res_df_ht_comb
comb_results['feat_imp_ht'] = feat_imp_ht_comb
comb_results['sensor_imp_ht'] = sensor_imp_ht_comb

comb_results['cv_res_pa'] = cv_res_df_pa_comb
comb_results['feat_imp_pa'] = feat_imp_pa_comb
comb_results['sensor_imp_pa'] = sensor_imp_pa_comb

imu_results = {}
imu_results['cv_res_mv'] = cv_res_df_mv_imu
imu_results['feat_imp_mv'] = feat_imp_mv_imu

imu_results['cv_res_ht'] = cv_res_df_ht_imu
imu_results['feat_imp_ht'] = feat_imp_ht_imu

imu_results['cv_res_pa'] = cv_res_df_pa_imu
imu_results['feat_imp_pa'] = feat_imp_pa_imu

stretch_results = {}
stretch_results['cv_res_mv'] = cv_res_df_mv_stretch
stretch_results['feat_imp_mv'] = feat_imp_mv_stretch

stretch_results['cv_res_ht'] = cv_res_df_ht_stretch
stretch_results['feat_imp_ht'] = feat_imp_ht_stretch

stretch_results['cv_res_pa'] = cv_res_df_pa_stretch
stretch_results['feat_imp_pa'] = feat_imp_pa_stretch

emg_results = {}
emg_results['cv_res_mv'] = cv_res_df_mv_emg
emg_results['feat_imp_mv'] = feat_imp_mv_emg

emg_results['cv_res_ht'] = cv_res_df_ht_emg
emg_results['feat_imp_ht'] = feat_imp_ht_emg

emg_results['cv_res_pa'] = cv_res_df_pa_emg
emg_results['feat_imp_pa'] = feat_imp_pa_emg


#%%

desired_filename = 'dict.emg_results'

pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/results/{}'.format(desired_filename),"wb")
pickle.dump(emg_results, pickle_out)
pickle_out.close()  

desired_filename = 'dict.imu_results'

pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/results/{}'.format(desired_filename),"wb")
pickle.dump(imu_results, pickle_out)
pickle_out.close()  

desired_filename = 'dict.stretch_results'

pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/results/{}'.format(desired_filename),"wb")
pickle.dump(stretch_results, pickle_out)
pickle_out.close()  

desired_filename = 'dict.comb_results'

pickle_out = open(r'/Users/Kieran/OneDrive - Nanyang Technological University/High-Level HMI/Experiment 1/Human_Motion_Intention_Analysis/results/{}'.format(desired_filename),"wb")
pickle.dump(comb_results, pickle_out)
pickle_out.close()  
        
#%%

angles = [2,5,10,15]

best_scores_mv = []
best_scores_ht = []
best_scores_pa = []

for i in range(0,len(angles)):

    best_scores_mv.append(cv_res_df_mv[i].loc[0].CrossValMeans)
    best_scores_ht.append(cv_res_df_ht[i].loc[0].CrossValMeans)
    best_scores_pa.append(cv_res_df_pa[i].loc[0].CrossValMeans)
    
plt.figure(1)
plt.plot(angles,best_scores_mv)
plt.plot(angles,best_scores_ht)
plt.plot(angles,best_scores_pa)











