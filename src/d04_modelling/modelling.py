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

#%% Read data

dataset = pd.read_csv(r'data/03_processed/data_processed')

#%% Train Test Split

X = dataset.drop(['peak_amplitude','peak_velocity','mean_velocity','time_end', 'time_half'],axis=1)
y = dataset[['peak_velocity','time_half','peak_amplitude','mean_velocity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

y_train_dur = y_train['time_half']
y_train_mv = y_train['mean_velocity']
    
y_test_dur = y_test['time_half']
y_test_mv = y_test['mean_velocity']
    

#%% Visualisations

g = sns.heatmap(X_train.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")


#%% Modelling

### Scaling

# Scale the train data to range [0 1] and scale the test data according to the train data
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_scaled = min_max_scaler.fit_transform(X_train)
# X_test_scaled = min_max_scaler.transform(X_test)

### Baseline regression models

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

baseline_regression_models(X_train, y_train_mv, kfolds)
baseline_regression_models(X_train, y_train_dur, kfolds)

#  Best baseline model for velocity: 
#     1. RidgeCV
#     2. LassoCV
#     3. ElasticNetCV
#     4. Gradiant Boosting
#     5. Random Forest

#  Best baseline model for duration: 
#     1. ExtraTrees
#     2. RandomForest
#     3. SVR
#     4. Adaboost
#     5. Gradient Boosting

#%% Hyper parameter tuning



#%%

ridge=Ridge()
parameters= {'alpha':[x for x in [1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1, 10]],'normalize':[True,False],"fit_intercept": [True, False], "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

ridge_reg=GridSearchCV(ridge, cv=kfolds, param_grid=parameters,scoring = "neg_mean_absolute_error",n_jobs=2)

ridge_reg.fit(X_train,y_train_mv)
print("The best value of Ridge is: ",ridge_reg.best_params_)
print("Score: " ,-ridge_reg.best_score_)


#%%
from sklearn.linear_model import Lasso

Lasso_reg =Lasso()
parameters= {'alpha':[0.015],'normalize':[True,False], "fit_intercept": [True, False],'max_iter':[5000], 'tol':[0.01,0.001,0.0001]}

Lasso_reg=GridSearchCV(Lasso_reg, cv=kfolds, param_grid=parameters,scoring = "neg_mean_absolute_error",n_jobs=2)
Lasso_reg.fit(X_train,y_train_mv)

print("The best value of Lasso is: ",Lasso_reg.best_params_)
print("Score: " ,-Lasso_reg.best_score_)

#%% Multiple Layer Perceptron

parameter_space = {
    'hidden_layer_sizes': [40,50,60,70],
    'activation': ['relu','tanh'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.005],
    'learning_rate': ['constant','adaptive'], 
    'random_state':[42]
}

MLPR = MLPRegressor()

MLPR = GridSearchCV(MLPR, parameter_space, n_jobs=-1, cv=kfolds, scoring = "neg_mean_absolute_error")
MLPR.fit(X_train, y_train_mv)

print("The best value of MLP is: ",MLPR.best_params_)
print("Score: " ,-MLPR.best_score_)



#%%



### Random Forest

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    # 'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap
                   'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
               
               }

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)
# Fit the random search model
rf_random.fit(X_train, y_train_mv)

RF_best =  rf_random.best_estimator_.feature_importances_

#%%
#columns = time, stretch1, BB1, TB1, AD1, P1, Pos1,Vel1, acc1 
-cross_val_score(rf_random.best_estimator_, X_train_scaled, y_train['Mean Velocity'].values, scoring = "neg_mean_absolute_error", cv = kfolds, n_jobs=1)


lst = []

for i in range(1,7):
    lst.append('time{}'.format(i))
    lst.append('stretch{}'.format(i))
    lst.append('BB{}'.format(i))
    lst.append('TB{}'.format(i))
    lst.append('AD{}'.format(i))
    lst.append('PC{}'.format(i))
    lst.append('Pos{}'.format(i))
    lst.append('Vel{}'.format(i))
    lst.append('Acc{}'.format(i))

series = pd.Series(lst)
indices = np.argsort(RF_best)[::-1][:20]

g = sns.barplot(y = series[indices], x = RF_best[indices])

#%%
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTR = DecisionTreeRegressor()

adaDTR = AdaBoostClassifier(DTR) #random_state=7)

ada_param_grid = {
              'n_estimators': (1, 2),
              'base_estimator__max_depth': (1, 2),
              'algorithm': ('SAMME', 'SAMME.R')}

gsadaDTR = GridSearchCV(adaDTR,param_grid = ada_param_grid, cv=kfolds, scoring="neg_mean_absolute_error", verbose = 1)

gsadaDTR.fit(X_train,y_train['Mean Velocity'])

ada_best = gsadaDTR.best_estimator_




#%%

nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

ada_best = 
ExtC_best = 
RFC_best = 
GBC_best = 


names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1

#%%

#full_combined_trajectories_test = full_combined_trajectories[full_combined_trajectories.keys()==X_test.index]
indices = X_test.index.tolist()

test_trajectories = [filtered_mj_traj[i] for i in filtered_mj_traj.keys() if i in indices] 
#evaluate(full_combined_features)    

#%% ANN

ANN(filtered_mj_feat_out)

#%%

filtered_mj_feat_out.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end', 'Time at half'],axis=1)

        
 #%% MAKE PREDICTIONS ON TEST DATA

#print('-----------Multiple Linear Regression---------------\n')
#X_train_scaled, X_test_scaled, y_train, y_test, df_predictions = MLR(extracted_features)
    
#print('------------Support Vector Regression----------------\n')
#clf,clf2,min_max_scaler, min_max_scaler2 = Support_Vector_Regression(extracted_features)

#df_predictions_SVM = predict_values(extracted_features_test[1],clf,clf2,min_max_scaler, min_max_scaler2)
    
#evaluate(df_predictions_SVM)    

#%%

current = test_trajectories[0].iloc[0,-3]
#setpoint = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-3]*df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]
setpoint = X_test.iloc[0,-1]*X_test.iloc[0,-2]*2

frequency = 100
time =X_test.iloc[0,-2]*2#(df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]-current)/df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-2]
        
traj1, traj_vel = mjtg(current, setpoint, frequency, time)

plt.plot(traj1)

#%% PLOT PREDICTED TRAJECTORIES

plot_predicted_traj(X_test,test_trajectories) #df_predictions_SVM
#plt.savefig('predicted_and_real_traj_40%.svg', format='svg')

#%%
plt.plot(test_trajectories[30]['time'], test_trajectories[30]['angular position'])
plt.savefig('real_traj_40%.svg', format='svg')

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









