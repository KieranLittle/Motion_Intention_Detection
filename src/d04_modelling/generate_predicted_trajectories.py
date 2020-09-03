#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:39:45 2020

@author: Kieran
"""

"""
Making predictions and then plot the predicted trajectory

"""
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