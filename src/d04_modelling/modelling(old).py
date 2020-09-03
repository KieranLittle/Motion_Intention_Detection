#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:36:21 2020

@author: Kieran
"""

"""

Tuning Models

"""


#%% Train Test Split

# X = dataset.drop(['peak_amplitude','peak_velocity','mean_velocity','time_end', 'time_half'],axis=1)

# #X = X_reduced #####CHANGE

# y = dataset[['peak_velocity','time_half','peak_amplitude','mean_velocity']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# y_train_dur = y_train['time_half']
# y_train_mv = y_train['mean_velocity']
# y_train_pa = y_train['peak_amplitude']
    
# y_test_dur = y_test['time_half']
# y_test_mv = y_test['mean_velocity']
# y_test_pa = y_test['peak_amplitude']

# mean_peak_velocity = y.peak_velocity.mean()
# mean_time_half = y.time_half.mean()
# mean_peak_amplitude = y.peak_amplitude.mean()
# mean_mean_velocity = y.mean_velocity.mean()


# #%% Visualisations

# g = sns.heatmap(X_train.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")


#%% FEATURE REDUCTION USING PCA?

# from sklearn.decomposition import PCA

# pca = PCA(0.99)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# pca.n_components_


#%% TUNING OF BEST MODELS TO CHECK IMPROVMENT
   
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report
# #from sklearn.grid_search import GridSearchCV 

# kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# X = dataset_dict[0].drop(['Peak Amplitude','Peak Velocity','Mean Velocity','T_end', 'T_half'],axis=1)
# y = dataset_dict[0][['Peak Amplitude','Peak Velocity','Mean Velocity', 'T_half']]
    
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
# y_train_dur = y_train['T_half']
# y_train_mv = y_train['Mean Velocity']
# y_train_pa = y_train['Peak Amplitude']
        
# y_test_dur = y_test['T_half']
# y_test_mv = y_test['Mean Velocity']
# y_test_pa = y_test['Peak Amplitude']

# mean_time_half = y.T_half.mean()
# mean_peak_amplitude = y['Peak Amplitude'].mean()
# mean_mean_velocity = y['Mean Velocity'].mean()

# baseline = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10)
# baseline.fit(X_train,y_train_mv)
# predictors=list(X_train)
# feat_imp = pd.Series(baseline.feature_importances_, predictors).sort_values(ascending=False)[:10]
# feat_imp.plot(kind='bar', title='Importance of Features')
# plt.ylabel('Feature Importance Score')
# print('Accuracy of the GBM on test set: {:.3f}'.format(baseline.score(X_test, y_test_mv)))
# pred=baseline.predict(X_test)
# print('Test Error = ',np.mean(abs(pred-y_test_mv))*100/mean_mean_velocity)

# cross_val_error = np.mean(-cross_val_score(baseline, X_train, y_train_mv, scoring = "neg_mean_absolute_error", cv = kfolds, n_jobs=n_jobs))*100/mean_mean_velocity

# print('Cross Validation Error =',cross_val_error)


# #%% Hyper parameter tuning

# GBR = GradientBoostingRegressor()

# GBR.fit(X_train, y_train_mv)

# plt.scatter(y_test_mv, GBR.predict(X_test))
# plt.plot([0,175],[0,175],'r')

# #%%

# ridge=Ridge()
# parameters= {'alpha':[x for x in [1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1, 10]],'normalize':[True,False],"fit_intercept": [True, False], "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

# ridge_reg=GridSearchCV(ridge, cv=kfolds, param_grid=parameters,scoring = "neg_mean_absolute_error",n_jobs=2)

# ridge_reg.fit(X_train,y_train_mv)
# print("The best value of Ridge is: ",ridge_reg.best_params_)
# print("Score: " ,-ridge_reg.best_score_)


# #%%
# from sklearn.linear_model import Lasso

# Lasso_reg =Lasso()
# parameters= {'alpha':[0.015],'normalize':[True,False], "fit_intercept": [True, False],'max_iter':[5000], 'tol':[0.01,0.001,0.0001]}

# Lasso_reg=GridSearchCV(Lasso_reg, cv=kfolds, param_grid=parameters,scoring = "neg_mean_absolute_error",n_jobs=2)
# Lasso_reg.fit(X_train,y_train_mv)

# print("The best value of Lasso is: ",Lasso_reg.best_params_)
# print("Score: " ,-Lasso_reg.best_score_)

# #%% Multiple Layer Perceptron

# parameter_space = {
#     'hidden_layer_sizes': [20,40,50,60,70],
#     'activation': ['relu','tanh'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.005],
#     'learning_rate': ['constant','adaptive'], 
#     'random_state':[42]
# }

# MLPR = MLPRegressor()

# MLPR = GridSearchCV(MLPR, parameter_space, n_jobs=-1, cv=kfolds, scoring = "neg_mean_absolute_error")
# MLPR.fit(X_train, y_train_mv)

# print("The best value of MLP is: ",MLPR.best_params_)
# print("Score: " ,-MLPR.best_score_)

# #%%
# y = y_test_dur
# a = np.min(y)
# b = np.max(y)


# plt.scatter(y_test_dur,MLPR.predict(X_test))
# #plt.plot([i for i in range(0.5,2.5)],[i for i in range(0.5,2.5)],'r')



#%% Random Forest

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
np.mean(-cross_val_score(rf_random.best_estimator_, X_train, y_train_mv, scoring = "neg_mean_absolute_error", cv = kfolds, n_jobs=1))

#series = pd.Series(lst)
indices = np.argsort(RF_best)[::-1][:10]

g = sns.barplot(y = X_train.columns[indices], x = RF_best[indices])
#g = sns.barplot(x = ['1','2','3','4','5'], y = RF_best[indices])

#%%
X_reduced = X[X.columns[indices]]



#%%
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING


# Adaboost
DTR = DecisionTreeRegressor()

adaDTR = AdaBoostRegressor(DTR) #random_state=7)
#adaDTR = AdaBoostClassifier(DTR) #random_state=7)

ada_param_grid = {
              'n_estimators': (1, 2),
              'base_estimator__max_depth': (1, 2),
              'algorithm': ('SAMME', 'SAMME.R')}

gsadaDTR = GridSearchCV(adaDTR,param_grid = ada_param_grid, cv=kfolds, scoring="neg_mean_absolute_error", verbose = 1)

gsadaDTR.fit(X_train,y_train_mv)

ada_best = gsadaDTR.best_estimator_




#%%

nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

ada_best = 
ExtC_best = 
RFC_best = RF_best
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
        
