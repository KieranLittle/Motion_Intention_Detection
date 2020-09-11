#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:17:40 2020

@author: Kieran
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing

from collections import Counter
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge, ElasticNet, Lasso
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

#%%
def read_file(sub_num, trial_num):
    
    """
    1. Read in the files using the subject number and the trial number
    2. Removes repeated values from the amplitudes command
    3. Labels axes and combines dataframes into one
    
    Input: 
        sub_num: subject number used to label data folder
        trial_num: the number of the trial
        
    Output:
        combined_dataset: data combined into one dataframe (for 1 trial, 1 subject)
    
    """
    
    import pandas as pd
    
    global amplitudes
    
    # Define path to the data
    path = 'data/01_raw/sub_'+sub_num
    
    # Read in the different files
    # (the stretch signal and the imu and combined)
    # emg_MVC (MVC = Maximum Voluntary Contraction): this file is used to normalise the muscle activity between subjects
    
    stretch_imu = pd.read_csv(path+'/imu/Trial'+trial_num+'.csv', header=None)
    emg = pd.read_csv(path+'/emg/Trial'+trial_num+'.csv', header=None)#'/emg/Trial4_emg.csv'
    amplitude = pd.read_csv(path+'/Trajectory.csv', header=None)
    emg_MVC = pd.read_csv(path+'/emg/MVC.csv', header=None)
    
    # the amplitude file column: 0 simulation time of the trial, column 1 = amplitude commanded to person
    
    # find the commands which are sent to the subject (they repeat for as long as it was shown to the user)
    amplitude2 = amplitude[amplitude.iloc[:,1] != 0]
    
    # create a new list to find the order of each command without repeats
    amplitudes = []
    amplitudes.append(amplitude2.iloc[0,1])
    
    # find if the next command is the same as the previous command
    for i in range(len(amplitude2)-1):
        if (amplitude2.iloc[i,1] != amplitude2.iloc[i+1,1]):
            amplitudes.append(amplitude2.iloc[i+1,1])
    
    # label columns of emg dataframe
    emg = emg.T
    emg.columns = ['Time','Biceps','Triceps','Deltoid','Pecs']
    
    # label columns of emg
    emg_MVC = emg_MVC.T
    emg_MVC.columns = ['Time','Biceps','Triceps','Deltoid','Pecs']

    # label columns of stretch and imu dataframe
    stretch_imu = stretch_imu.T
    stretch_imu.columns = ['time','imu','stretch']
    
    # combine the dataframes into one 
    combined_dataset = stretch_imu
    combined_dataset['Biceps'] = emg['Biceps']
    combined_dataset['Triceps'] = emg['Triceps']
    combined_dataset['Deltoid'] = emg['Deltoid']
    combined_dataset['Pecs'] = emg['Pecs']
    
    return combined_dataset, amplitudes, emg_MVC

def resample(dataset):
    
    """
    This function downsamples the signals in the original signals to 100 Hz.
    
    """
    
    import scipy
    import numpy as np
    from scipy import signal
    import pandas as pd
    
    #a = dataset[dataset['time'] == 1200.0].index
    #dataset.drop(a , inplace=True)
    
    # resamples the signals to 100Hz:
    
    dataset = dataset.loc[:200000];
    
    resampled_stretch = scipy.signal.resample(dataset.iloc[:,2],120001)
    resampled_imu = scipy.signal.resample(dataset.iloc[:,1],120001)
    resampled_BB = scipy.signal.resample(dataset.iloc[:,3],120001)
    resampled_TB = scipy.signal.resample(dataset.iloc[:,4],120001)
    resampled_D = scipy.signal.resample(dataset.iloc[:,5],120001)
    resampled_P = scipy.signal.resample(dataset.iloc[:,6],120001)
    
    # Define new time signal
    time = np.linspace(0,12000*0.1,120001)

    dataset = pd.DataFrame({'time': time, 'stretch': resampled_stretch, 'imu': resampled_imu, 'biceps': resampled_BB,\
                            'triceps': resampled_TB, 'deltoid': resampled_D, 'pecs': resampled_P})
    
    # rounds away floating point inaccuracies
    decimals = 2 
    dataset['time'] = dataset['time'].apply(lambda x: round(x, decimals))
    dataset['imu'] = dataset['imu'].apply(lambda x: round(x, decimals))
    
    return dataset
    

def filteremg(time, emg, low_pass=49, sfreq=100, high_band=2, low_band=40, graphs=1):
    import scipy as sp
    import matplotlib.pyplot as plt
    
    
    """
    Create bandpass filter to filter emg, returns filtered emg signal
    
    time: time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    
    # # normalise cut-off to sfreq
    high_band = high_band/(sfreq/2)
    low_band = low_band/(sfreq/2)
    
    # # create bandpass filter
    b1, a1 = sp.signal.butter(2, [high_band,low_band], btype='bandpass')
    
    # # apply forward backward filtering filtfilt
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)    
    
    # # rectify
    emg_rectified = abs(emg_filtered)
    #emg_rectified = abs(emg)
    
    # # apply low pass filter to rectified signal
    low_pass = low_pass/sfreq
    
    #high_pass = 20/sfreq
    
    b2, a2 = sp.signal.butter(2, low_pass, btype='lowpass')
    #b2, a2 = sp.signal.butter(4, high_pass, btype='highpass')
    
    
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)
    
    if graphs == 1:
        
        range_slt = 400
        plt.subplot(1,2,1)
        plt.plot(time[0:range_slt],emg[0:range_slt])
        
        plt.subplot(1,2,2)
        plt.plot(time[0:range_slt],emg_envelope[0:range_slt])
        
        plt.show()

    return emg_envelope

def lowpass_filter(time, signal, order=1, low_pass=10, sfreq=100):
    
    import scipy as sp
    import matplotlib.pyplot as plt
    
    """ 
    create a low pass filter
    
    """
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/sfreq
    b, a = sp.signal.butter(order, low_pass, btype='lowpass')
    filtered_signal = sp.signal.filtfilt(b, a, signal) #imu_rectified)
    
    plt.figure(1, figsize=(15,8))
    plt.subplot(2,5,1)
    plt.plot(time[0:10000],signal[0:10000])
    plt.subplot(2,5,2)
    plt.plot(time[0:10000],filtered_signal[0:10000])
    plt.subplot(2,5,3)
    plt.plot(time[0:1000],signal[0:1000])
    plt.subplot(2,5,4)
    plt.plot(time[0:1000],filtered_signal[0:1000])
    plt.subplot(2,5,5)
    plt.plot(time[0:1000],signal[0:1000])
    plt.plot(time[0:1000],filtered_signal[0:1000],'r')
    plt.show()
    
    return filtered_signal

def filter_derivate(dataset):
    
    """
    1. filter the imu and emg signals in the dataset
    2. defines the differentials of the imu signal to find velocity and acceleration
    3. filters the emg signals and finds the derivatives
    
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # define new dataset based on the input dataset
    new_dataset = dataset.copy()
    
    # filter the imu signal and relabel as the angular position
    new_dataset['angular position'] = lowpass_filter(new_dataset['time'],new_dataset['imu'],10,3,100)
    
    # drop the old column called 'imu'
    new_dataset.drop('imu',axis=1,inplace=True)
    
    # define the angular velocity based on the difference between points divided by sampling time
    vel = np.diff(new_dataset['angular position'],1)/0.01
    
    # add a last value to the end to make lengths equal
    vel = np.append(vel,0)
    
    # filter the angular velocity signal
    new_dataset['angular velocity'] = lowpass_filter(new_dataset['time'],vel,2,3,100)
    
    # define the angular acceleration based on the change in velocity divded by sampling time
    acc = np.diff(new_dataset['angular velocity'],1)/0.01
    
    # add 0 to make lengths equal
    acc = np.append(acc,0)
    
    # filter angular acceleration 
    new_dataset['angular acceleration'] = lowpass_filter(new_dataset['time'],acc,2,3,100)
   
    # define the stretch derivatives and filter
    new_dataset['stretch'] = lowpass_filter(new_dataset['time'],new_dataset['stretch'],2,4,100)
    new_dataset['stretch_d'] = np.append(np.diff(new_dataset['stretch'],1)/0.01,0)
    new_dataset['stretch_dd'] =np.append(np.diff(new_dataset['stretch_d'],1)/0.01,0)
    
    # filter emg
    biceps_filtered = filteremg(new_dataset['time'],new_dataset['biceps'])
    triceps_filtered = filteremg(new_dataset['time'],new_dataset['triceps'])
    deltoid_filtered = filteremg(new_dataset['time'],new_dataset['deltoid'])
    pecs_filtered = filteremg(new_dataset['time'],new_dataset['pecs'])
    
    new_dataset['biceps']=biceps_filtered
    new_dataset['triceps']=triceps_filtered
    new_dataset['deltoid']=deltoid_filtered
    new_dataset['pecs']=pecs_filtered
    
    # find the derivative for emg signals
    new_dataset['biceps_d']=np.append(np.diff(new_dataset['biceps'],1)/0.01,0)
    new_dataset['triceps_d']=np.append(np.diff(new_dataset['triceps'],1)/0.01,0)
    new_dataset['deltoid_d']=np.append(np.diff(new_dataset['deltoid'],1)/0.01,0)
    new_dataset['pecs_d']=np.append(np.diff(new_dataset['pecs'],1)/0.01,0)
    
    new_dataset['biceps_dd']=np.append(np.diff(new_dataset['biceps_d'],1)/0.01,0)
    new_dataset['triceps_dd']=np.append(np.diff(new_dataset['triceps_d'],1)/0.01,0)
    new_dataset['deltoid_dd']=np.append(np.diff(new_dataset['deltoid_d'],1)/0.01,0)
    new_dataset['pecs_dd']=np.append(np.diff(new_dataset['pecs_d'],1)/0.01,0)

    return new_dataset

def segment_data(dataset, angle_cutoff = [2,5,10,15]):
    
    import numpy as np
    import pandas as pd
    
    """
    Split the trajectories in dataset into segments with different window lengths (defined by different cut-off elbow angles)
    
    """
    
    # define window lengths
    
    # intialise variables
    count = 0
    amplitude_temp = []
    amplitude_change =[]
    half_amplitude_change = []
    
    start_angle = []
    traj_segment ={}
    trajectory = {}
    trajectory_temp = {}
    segments = {}
    
    time_start = []
    time_half = []
    time_end = []
    
    peak_segment_velocity = []
    mean_segment_velocity = []
    segment_max_point = []
    segment_cut_off = []
    
    # initialise a dictionary for each segment
    
    for j in range(0,len(angle_cutoff)):
        segments[j] = {}

    # loop over the dataset and locate the trajectory and then segment that trajectory into segments 
    
    for j in range(0,len(dataset['time'])):
        
        # there is nothing in the first 8 seconds and the dataset finishes at 12000.0 so ignore these
      if (dataset['time'][j] % 8.0 == 0.0) & (dataset['time'][j] != 0.0) & (dataset['time'][j] != 1200.0): #& (dataset['imu'][j] < 5):
          
          # Find the trajectory in the the window 0:7 seconds so make a first split between X and X+ 7 seconds
          trajectory_temp[count] = dataset.iloc[j:j+700,:] #400
          
          # define the angular position and angular velocity columns
          ap_col = trajectory_temp[count].columns.get_loc("angular position")
          av_col = trajectory_temp[count].columns.get_loc("angular velocity")
          
          """
          Find the end of the trajectory
          """
          # find the index when the angular position is < max- (max_threshold)
          max_threshold = 5.0
          f = trajectory_temp[count].loc[((trajectory_temp[count].iloc[:,ap_col] > np.max(trajectory_temp[count].iloc[:,ap_col]-max_threshold)))] #& (trajectory_temp[count].iloc[:,-2]>0))] 
          end = f.index[0]- dataset.index[j]  
                                     
          # Define the end trajectory period to search for the end of the motion
          end_traj = trajectory_temp[count][(end-50):]
          
          # search within the end of the trajectory to find when the velocity drops below 1+ min velocity
          
          # set angular velocity threshold
          end_velocity_threshold = 1.0
          
          # find the index when the velocity is below the threshold and set this as the end of the trajectory
          amp_range = end_traj.loc[(end_traj['angular velocity'] <  end_velocity_threshold+min(abs(end_traj.iloc[:,-2])))] #0.99*max(end_traj.iloc[:,-3]))] 
          end_amplitude_index = amp_range.index[0]
          
          """
          Find the beginning of the trajectory
          """
          
          # define the velocity threshold of the beginning of the segment
          start_velocity_threshold = 1.0
          
          # find the index where the angular velocity exceeds the threshold
          a = trajectory_temp[count].loc[(trajectory_temp[count].iloc[:,av_col] > 1.0)]   #&(trajectory[count]['angular velocity']>0)]  
          start_amplitude_index = a.index[0]
          
          # Define the trajectory based on the start and end index that we have previously found and see if it works:        
          trajectory_temp[count] = dataset.iloc[start_amplitude_index:end_amplitude_index,:]#end_amplitude_index,:]
          
          # This was to adapt the start and end index
          # mean_velocity_temp = np.mean(trajectory_temp[count]['angular velocity'])  
          # start_amplitude_index = start_amplitude_index_temp  #-int((mean_velocity_temp//2))
          # end_amplitude_index = end_amplitude_index   #+int((mean_velocity_temp//2))
          
          """
          Define the final trajectory
          """
          # define the final trajectory
          trajectory[count] = dataset.iloc[start_amplitude_index:end_amplitude_index,:]
          
          
          """
          Create segments based on different window lengths
          """
          
          
          # Find the start and end angular position
          end_amplitude =  trajectory[count].iloc[-1,ap_col] #trajectory[count].iloc[amp_range_index,-3]
          start_amplitude = trajectory[count].iloc[0,ap_col]
          
         
          # for each loop create a segment that is from the start to different angle cut-offs
          
          for angle in angle_cutoff:
          
              cut_off_index = trajectory[count].loc[(trajectory[count].iloc[:,ap_col] < (start_amplitude + angle))].index[-1]
              segments[angle_cutoff.index(angle)][count] = dataset.iloc[start_amplitude_index:cut_off_index,:]
          
          """
          Extract information about the trajectory that will be predicted in the learning phase
          """
        
          # define the change in ampltiude from the start to the end
          amplitude_change_ = end_amplitude- start_amplitude
          amplitude_change.append(amplitude_change_)
          
          # Find the time at halfway to the end of the trajectory
          d = trajectory[count].loc[((trajectory[count]['angular position'] - start_amplitude) < (amplitude_change_/2))]
          half_time_ = d.iloc[-1,0]- trajectory[count].iloc[0,0]
          amplitude_half = d.iloc[-1,ap_col] #- start_amplitude
          
          # Append half time and half amplitude to list
          time_half.append(half_time_)
          half_amplitude_change.append(amplitude_half)
          
          # Find peak and mean velocity and append to lists
          peak_velocity = np.max(trajectory[count]['angular velocity'])   #np.max(np.diff(dataset[j:j+400]['imu'],1))/0.01
          #mean_velocity = np.mean(trajectory[count]['angular velocity'])  
          mean_velocity = amplitude_half/half_time_
          
          peak_segment_velocity.append(peak_velocity)
          mean_segment_velocity.append(mean_velocity)
          
          # append the start and end index of trajectory
          segment_cut_off.append(start_amplitude_index)
          segment_max_point.append(end_amplitude_index)
          
          # Find the start position
          start_angle.append(start_amplitude)
          
          # time of start and end trajectory
          time_start_ = trajectory[count].iloc[0,0]
          time_end_ = trajectory[count].iloc[-1,0]-time_start_
          
          #time_start.append(time_start_)
          time_end.append(time_end_)
          
          count=count+1
    
    # Round the values to avoid precision 
    peak_segment_velocity = [ round(elem, 2) for elem in peak_segment_velocity ]
    time_end = [ round(elem, 2) for elem in time_end ]
    time_half = [ round(elem, 2) for elem in time_half ]

    # create a dataframe from the values extracted from the trajectories 
    dict = {'amplitude_change': amplitude_change, 'peak_velocity': peak_segment_velocity, 'mean_velocity': mean_segment_velocity, \
            'time_half':time_half, 'time_end':time_end, 'segment_cut_off':segment_cut_off, 'segment_max_point':segment_max_point, 'start_angle':start_angle}
        
    df = pd.DataFrame(dict)
        
    return segments, trajectory, df

def plot_segments(segments, peak_amplitude,peak_velocity):
    
    import matplotlib.pyplot as plt
    
    for i in range(0,20):
        
        plt.figure(i,figsize=(15,4)) 
        
        #plt.subtitle('Peak Velocity: ' +str(peak_velocity[i]) +' , Peak Amplitude: '+ str(peak_amplitude[i]))
        plt.subplot(1,6,1)
        plt.plot(segments[i]['time'],segments[i]['angular position'])
        #plt.ylim(0,10)
        
        plt.subplot(1,6,2)
        plt.plot(segments[i]['time'],segments[i]['angular velocity'])
        #plt.ylim(0,10)
        
        plt.subplot(1,6,3)
        plt.plot(segments[i]['time'],segments[i]['angular acceleration'])
        #plt.ylim(0,10)
        
        plt.subplot(1,6,4)
        plt.plot(segments[i]['time'],segments[i]['stretch'])
#        plt.ylim(3.324,3.32)
        
        plt.subplot(1,6,5)
        plt.plot(segments[i]['time'],segments[i]['biceps'])
        #plt.ylim(0.005,0.015)
        
        plt.subplot(1,6,6)
        plt.plot(segments[i]['time'],segments[i]['triceps'])
        #plt.ylim(0.003,0.008)
        
        # plt.subplot(1,6,5)
        # plt.plot(segments[i]['time'],segments[i]['deltoid'])
        # #plt.ylim(0.002,0.004)
        
        # plt.subplot(1,6,6)
        # plt.plot(segments[i]['time'],segments[i]['pecs'])
        # #plt.ylim(0.003,0.005)
        
        plt.show()
    
    return 0

def plot_traj(trajectory,segment_line, segment_max_point):
    
    import matplotlib.pyplot as plt
    
    
    for segment_num in range(0,20):
        
        segment_line[segment_num] = segment_line[segment_num] + trajectory[segment_num].iloc[0,0]
        segment_max_point[segment_num] = segment_max_point[segment_num] + trajectory[segment_num].iloc[0,0]
        
        
        #plt.plot(trajectory[segment_num]['time']-trajectory[segment_num].iloc[0,0],segment_line[segment_num])
        
        plt.figure(segment_num,figsize=(15,4)) 
        #plt.suptitle('Peak Velocity: ' +str(peak_velocity[segment_num]) +' , Peak Amplitude: '+ str(peak_amplitude[segment_num]))
        
        time = trajectory[segment_num]['time']#-trajectory[segment_num].iloc[0,0]
        
        plt.subplot(1,6,1)
        plt.plot(time,trajectory[segment_num]['angular position'])
        plt.title('segment cuff off: {} and trajectory max point = {}'.format(segment_line[segment_num], segment_max_point[segment_num]))
        
        plt.subplot(1,6,2)
        plt.plot(time,trajectory[segment_num]['angular velocity'])
        
        plt.subplot(1,6,3)
        plt.plot(time,trajectory[segment_num]['angular acceleration'])
        
        plt.subplot(1,6,4)
        plt.plot(time,trajectory[segment_num]['stretch'])
#        plt.ylim(3.324,3.32)
        
        plt.subplot(1,6,5)
        plt.plot(time,trajectory[segment_num]['biceps'])
        #plt.ylim(0.005,0.015)
        
        plt.subplot(1,6,6)
        plt.plot(time,trajectory[segment_num]['triceps'])
        #plt.ylim(0.003,0.008)
        
        # plt.subplot(1,6,5)
        # plt.plot(time,trajectory[segment_num]['deltoid'])
        # #plt.ylim(0.002,0.004)
        
        # plt.subplot(1,6,6)
        # plt.plot(time,trajectory[segment_num]['pecs'])
        #plt.ylim(0.003,0.005)
        
        
        plt.show()
        
    return 0
    

def extract_features(segments, df,filename,trial_num):
    
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  from scipy import stats
  
  #peak_amplitude, peak_velocities,  mean_segment_velocity,time_half,time_end
    
  ## Function to extract features from each segment

  #   Features: 1. total stretch
  #             2. total imu
  #             3. sum of biceps
  #             4. sum of tricep
  #             5. etc

  # stretch_sum = []
  # imu_sum = []
  # biceps_sum = []
  # triceps_sum = []
  # deltoid_sum = []
  # pecs_sum = []
  # #amplitude =[]

  # biceps_max = []
  # triceps_max = []
  # deltoid_max = []
  
  # stretch_max = []
  # imu_max = []
  # velocity_max = []
  
  # curr_velocity = []
  # curr_acc = []
  # curr_biceps_change = []
  # curr_biceps_change2 = []
  # curr_triceps_change = []
  # curr_triceps_change2 = []
  # curr_stretch_change = []
  # curr_stretch_change2 = []

  # biceps_diff = []
  # triceps_diff = []
  # deltoid_diff = []
  # stretch_diff = []
  # imu_diff = []
  
  # stretch_gradient = []
  # biceps_gradient = []
  # triceps_gradient = []
  # deltoid_gradient = []
  # pecs_gradient = []
  
  # biceps_min = []
  # biceps_min_time = []
  
  # triceps_min = []
  # deltoid_min = []
  
  segment_duration = []
  features_df = {} #pd.DataFrame()
  df_selected_features = {}
  
  # New features
  
  # stretch_mean = []
  # bb_mean = []
  # tb_mean =[]
  # ad_mean = []
  # pm_mean =[]
  
  # stretch_var = []
  # bb_var = []
  # tb_var = []
  # ad_var = []
  # pm_var = []
  
  col = []

  for i in range(1,7):
     col.append('time{}'.format(i))
     col.append('stretch{}'.format(i))
     col.append('bb{}'.format(i))
     col.append('tb{}'.format(i))
     col.append('ad{}'.format(i))
     col.append('pm{}'.format(i))
     col.append('pos{}'.format(i))
     col.append('vel{}'.format(i))
     col.append('acc{}'.format(i))
      
     col.append('stretch_d{}'.format(i))
     col.append('stretch_dd{}'.format(i))
        
     col.append('bb_d_{}'.format(i))
     col.append('tb_d_{}'.format(i))
     col.append('ad_d_{}'.format(i))
     col.append('pm_d_{}'.format(i))
     col.append('bb_dd_{}'.format(i))
     col.append('tb_dd_{}'.format(i))
     col.append('ad_dd_{}'.format(i))
     col.append('pm_dd_{}'.format(i))
     
  for window in range(0,len(segments)):
      
      features_df[window] = pd.DataFrame()
  
      for segment_number in range(0,len(segments[window])):
          
          segment_ind = segments[window][segment_number]
          
      #   ## Time
      #     segment_duration.append(segment_ind.iloc[-1,0] - segment_ind.iloc[0,0])
          
      #   ## Position  
          
      #     imu_sum.append(segment_ind['angular position'].sum())
      #     imu_max.append(np.max(segment_ind['angular position']))
      #     imu_diff.append(segment_ind.iloc[-1]['angular position'] - segment_ind.iloc[0]['angular position'])
          
      #   ## Velocity
      #     velocity_max.append(np.max(segment_ind['angular velocity']))
      #     curr_velocity.append(np.mean(segment_ind.iloc[-30:,-2]))
          
      #   ## Acceleration
      #     curr_acc.append(np.mean(segment_ind.iloc[-30:,-1]))#[(len(segments[segment_number])-30):]))
          
      #   ## Stretch
      #     stretch_sum.append(segment_ind['stretch'].sum())
      #     stretch_max.append(np.max(segment_ind.iloc[:-10,1]))
      #     curr_stretch_change.append(np.mean(np.diff(segment_ind['stretch'],1)[(len(segment_ind)-30):]))
      #     curr_stretch_change2.append(np.mean(np.diff(segment_ind['stretch'],2)[(len(segment_ind)-30):]))
      #     stretch_diff.append(segment_ind.iloc[-1]['stretch'] - segment_ind.iloc[0]['stretch'])
      #     #stretch_gradient.append(np.max(np.diff(segments[segment_number].iloc[:-10,1],1)/0.01))
          
      #   ## Biceps
          
      #     biceps_sum.append(segment_ind['biceps'].sum())
      #     biceps_max.append(np.max(np.abs(segment_ind['biceps'])))
      #     curr_biceps_change.append(np.mean(np.diff(segment_ind['biceps'],1)[(len(segment_ind)-30):]))
      #     curr_biceps_change2.append(np.mean(np.diff(segment_ind['biceps'],2)[(len(segment_ind)-30):]))
      #     #biceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,3],1)/0.01))
      #     biceps_diff.append(segment_ind.iloc[-1]['biceps'] - segment_ind.iloc[0]['biceps'])
      #     biceps_min.append(np.min(segment_ind['biceps']))
      #     biceps_min_time.append(segment_ind.loc[segment_ind.idxmin()['biceps']]['time']-segment_ind.iloc[0,0])
           
      #   ## Triceps
      #     triceps_sum.append(segment_ind['triceps'].sum())
      #     triceps_max.append(np.max(np.abs(segment_ind['triceps'])))
      #     curr_triceps_change.append(np.mean(np.diff(segment_ind['triceps'],1)[(len(segment_ind)-30):]))
      #     curr_triceps_change2.append(np.mean(np.diff(segment_ind['triceps'],2)[(len(segment_ind)-30):]))
      #     #triceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,4],1)/0.01))
      #     triceps_diff.append(segment_ind.iloc[-1]['triceps'] - segment_ind.iloc[0]['triceps'])
      #     triceps_min.append(np.min(segment_ind['triceps']))
        
      #   ## Deltoid
      #     deltoid_sum.append(segment_ind['deltoid'].sum())
      #     deltoid_max.append(np.max(np.abs(segment_ind['deltoid'])))
      #     #deltoid_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,5],1)/0.01))
      #     deltoid_diff.append(segment_ind.iloc[-1]['deltoid'] - segment_ind.iloc[0]['deltoid'])
      #     deltoid_min.append(np.min(segment_ind['deltoid']))
          
      #   ## Pecs
      #     pecs_sum.append(segment_ind['pecs'].sum())
      #     #pecs_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,6],1)/0.01))
         
      #     """ 
      #     Define segment length, split into 6 evenly spaced points
      #     Features of each point for each sensor (6):
      #       1. time
      #       2. position
      #       3. velocity 
      #       4. acceleration
      #       = 24 features
            
      #       + mean
      #       + var
      #     """
          
      #     stretch_mean.append(np.mean(segment_ind['stretch']))
      #     bb_mean.append(np.mean(segment_ind['biceps']))
      #     tb_mean.append(np.mean(segment_ind['triceps']))
      #     ad_mean.append(np.mean(segment_ind['deltoid']))
      #     pm_mean.append(np.mean(segment_ind['pecs']))
          
      #     stretch_var.append(np.var(segment_ind['stretch']))
      #     bb_var.append(np.var(segment_ind['biceps']))
      #     tb_var.append(np.var(segment_ind['triceps']))
      #     ad_var.append(np.var(segment_ind['deltoid']))
      #     pm_var.append(np.var(segment_ind['pecs']))
        
          segment_ind['time'] = np.round([i - segment_ind['time'].loc[0] for i in segment_ind['time']],2)
        
          segment_length = len(segment_ind) # in samples
          indicies = np.round(np.linspace(0,segment_length-1, 6)).astype(int)
          features_df_ravel = pd.DataFrame(np.array(segment_ind.iloc[indicies,:]).ravel())
          features_df_single = features_df_ravel.T
          features_df_single.columns = col
          
          features_df_single['Peak Amplitude'] = df['amplitude_change'][segment_number] #np.array(df['amplitude_change'])
          features_df_single['Peak Velocity'] = df['peak_velocity'][segment_number]
          features_df_single['Mean Velocity'] = df['mean_velocity'][segment_number]
          features_df_single['T_end'] = df['time_end'][segment_number]
          features_df_single['T_half'] = df['time_half'][segment_number]
          features_df_single['Subject_Num'] = int(filename)
          features_df_single['Trial_Num'] = int(trial_num)
        
          features_df[window] = features_df[window].append(features_df_single)
              
        
      # dict = {'curr_triceps_change2': curr_triceps_change2, 'curr_biceps_change2': curr_biceps_change2,\
      #         'triceps_min':triceps_min, 'triceps_max':triceps_max,'pecs_sum':pecs_sum,\
      #         'biceps_sum':biceps_sum,'biceps_max':biceps_max, 'biceps_min':biceps_min, \
      #         'curr_velocity':curr_velocity, 'curr_acc':curr_acc, \
      #         'segment_duration': segment_duration,'deltoid_sum':deltoid_sum,\
      #         'stretch_max':stretch_max}
          
   
      # 'stretch_gradient':stretch_gradient,
      
      #'biceps_min':biceps_min,'biceps_sum':biceps_sum, 'curr_biceps_change2': curr_biceps_change2, 'curr_triceps_change2': curr_triceps_change2} 
      #'window_max_velocity':velocity_max, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc, 'biceps_diff':biceps_diff, 'imu_diff':imu_diff, 'biceps_min':biceps_min, 'biceps_min_time':biceps_min_time } #{'biceps_sum':biceps_sum, 'biceps_max':biceps_max, 'biceps_gradient':biceps_gradient, 'triceps_gradient':triceps_gradient,'stretch_gradient':stretch_gradient, 'imu_diff':imu_diff, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc }
      #'deltoid_gradient':deltoid_gradient, 'pecs_gradient':pecs_gradient, 'imu_max':imu_max, 'stretch_gradient':stretch_gradient,'stretch_max':stretch_max} # 'stretch_gradient':stretch_gradient, 'window_max_velocity':velocity_max'stretch_gradient':stretch_gradient 'imu_diff':imu_diff 'window_max_velocity':velocity_max}  #'stretch_diff':stretch_gradient #stretch_diff':stretch_diff} #'velocity_max':velocity_max, 'biceps_max':biceps_max, 'triceps_diff':triceps_diff, 'deltoid_diff':deltoid_diff 'velocity_max':velocity_max 'stretch_diff':stretch_diff,'imu_diff':imu_diff,  'triceps_sum':triceps_sum,'deltoid_sum':deltoid_sum,'pecs_sum':pecs_sum,'amplitude':amplitude}
      
      # df_selected_features[window] = pd.DataFrame(dict) 
      
  # features_df['stretch_mean'] = stretch_mean
  # features_df['bb_mean'] = bb_mean
  # features_df['tb_mean'] = tb_mean
  # features_df['ad_mean'] = ad_mean
  # features_df['pm_mean'] = pm_mean

  # features_df['stretch_var'] = stretch_var
  # features_df['bb_var'] = bb_var
  # features_df['tb_var'] = tb_var
  # features_df['ad_var'] = ad_var
  # features_df['pm_var'] = pm_var
      


  #df2 = features_df.copy()
  
  # df2.columns = col
      
  #   #df2['End Window Time'] = time_3degrees
       
  #   #df['peak_segment_velocity'] = peak_segment_velocity
  # df2['Peak Amplitude'] = np.array(df['amplitude_change'])
  # df2['Peak Velocity'] = np.array(df['peak_velocity'])
  # df2['Mean Velocity'] = np.array(df['mean_velocity'])
    
  #   #df['Time at half traj'] = time_half
  # df2['Time at end'] = np.array(df['time_end'])
  # df2['Time at half'] = np.array(df['time_half'])
  # df2 = df2.reset_index(drop=True)
      
  #   #df2 = df2[(np.abs(stats.zscore(df2)) < 2).all(axis=1)]
    
  # #sns.pairplot(df2)
  # #plt.show()
      
  return features_df #df_selected_features,col


def segment_extract_features(segments, df, filename, trial_num, num_of_extracted_points = 6):
    
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  from scipy import stats
  
  
  # initialise variables
  segment_duration = []
  features_df = {} 
  df_selected_features = {}
  
  
  # define new column names for the new dataframe containing the extracted points 
  col = []

  for i in range(1,num_of_extracted_points+1):
     col.append('time{}'.format(i))
     col.append('stretch{}'.format(i))
     col.append('bb{}'.format(i))
     col.append('tb{}'.format(i))
     col.append('ad{}'.format(i))
     col.append('pm{}'.format(i))
     col.append('pos{}'.format(i))
     col.append('vel{}'.format(i))
     col.append('acc{}'.format(i))
      
     col.append('stretch_d{}'.format(i))
     col.append('stretch_dd{}'.format(i))
        
     col.append('bb_d_{}'.format(i))
     col.append('tb_d_{}'.format(i))
     col.append('ad_d_{}'.format(i))
     col.append('pm_d_{}'.format(i))
     col.append('bb_dd_{}'.format(i))
     col.append('tb_dd_{}'.format(i))
     col.append('ad_dd_{}'.format(i))
     col.append('pm_dd_{}'.format(i))
    
  # segments has the structure segments[window_length][segment_number]
  # for each segment[window_length] create a dataframe a new dataframe 
  
  for window in range(0,len(segments)):
      
      features_df[window] = pd.DataFrame()
      
      # for each of the segments in segments[window] 
  
      for segment_number in range(0,len(segments[window])):
          
          # select a segment
          segment_ind = segments[window][segment_number]
          
     
          """ 
           Define segment length, split into X evenly spaced points
           Features of each point for each sensor (6):
             X = num_of_extracted_points 
              
          """
          
      #     stretch_mean.append(np.mean(segment_ind['stretch']))
      #     bb_mean.append(np.mean(segment_ind['biceps']))
      #     tb_mean.append(np.mean(segment_ind['triceps']))
      #     ad_mean.append(np.mean(segment_ind['deltoid']))
      #     pm_mean.append(np.mean(segment_ind['pecs']))
          
      #     stretch_var.append(np.var(segment_ind['stretch']))
      #     bb_var.append(np.var(segment_ind['biceps']))
      #     tb_var.append(np.var(segment_ind['triceps']))
      #     ad_var.append(np.var(segment_ind['deltoid']))
      #     pm_var.append(np.var(segment_ind['pecs']))
        
          # make the start time of each segment = 0, and round all the time within the segment
          segment_ind['time'] = np.round([i - segment_ind['time'].iloc[0] for i in segment_ind['time']],2)
        
          # find the legnth of the segment
          segment_length = len(segment_ind) # in samples
          
          # create an array of X numbers between 0 and the segment length (X = num_of_extracted_points)
          indicies = np.round(np.linspace(0,segment_length-1, num_of_extracted_points)).astype(int)
          
          # extract the X number of points from the segment and then flatten the points to a 1D array
          features_df_ravel = pd.DataFrame(np.array(segment_ind.iloc[indicies,:]).ravel())
          
          # transpose and label the column names
          features_df_single = features_df_ravel.T
          features_df_single.columns = col
          
          # add extra details to the dataframe
          features_df_single['Peak Amplitude'] = df['amplitude_change'][segment_number] #np.array(df['amplitude_change'])
          features_df_single['Peak Velocity'] = df['peak_velocity'][segment_number]
          features_df_single['Mean Velocity'] = df['mean_velocity'][segment_number]
          features_df_single['T_end'] = df['time_end'][segment_number]
          features_df_single['T_half'] = df['time_half'][segment_number]
          features_df_single['Subject_Num'] = int(filename)
          features_df_single['Trial_Num'] = int(trial_num)
          
          # append the created row to a dataframe 
          features_df[window] = features_df[window].append(features_df_single)
                      
  return features_df


def MLR(df):

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    from sklearn import metrics
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    ##### Train model
    X_org = df.drop(['Peak Amplitude','Peak Velocity'],axis=1)
    
    #y = df['amplitude']
    y = df['Peak Velocity']
    
    # Split the data into train and test randomly with the test size = 30%, stratify data to split classification evenly
    X_train, X_test, y_train, y_test = train_test_split(X_org, y, test_size=0.50, random_state=42)
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    
    lm = LinearRegression()
    lm.fit(X_train_scaled,y_train)
    
    # The coefficients
    print('Coefficients: \n', lm.coef_)
    
    predictions = lm.predict(X_test_scaled)
    df_predictions = pd.DataFrame({'y_test':y_test, 'predictions':predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    
    X_scaled = min_max_scaler.transform(X_org)
    velocity_prediction = lm.predict(X_scaled)
    
    X = X_org.copy()
    X['velocity_prediction'] = velocity_prediction
    
    #X = df.drop(['Peak Amplitude','Peak Velocity'],axis=1)
    
    #y = df['amplitude']
    y = df['Peak Amplitude']
    
    # Split the data into train and test randomly with the test size = 30%, stratify data to split classification evenly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler2 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler2.fit_transform(X_train)
    X_test_scaled = min_max_scaler2.transform(X_test)
    
    lm2 = LinearRegression()
    lm2.fit(X_train_scaled,y_train)
    
    # The coefficients
    print('Coefficients: \n', lm.coef_)
    
    amplitude_predictions = lm2.predict(X_test_scaled)
    df_amplitude_predictions = pd.DataFrame({'y_test':y_test, 'predictions':amplitude_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_amplitude_predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('MAE:', metrics.mean_absolute_error(y_test, amplitude_predictions))
    print('MSE:', metrics.mean_squared_error(y_test, amplitude_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, amplitude_predictions)))
    
    #X_original_scaled = min_max_scaler.transform(X_original)
    
    scaled_velocity = min_max_scaler.transform(X_org)
    scaled_amplitude = min_max_scaler2.transform(X)
    
    velocity_prediction = lm.predict(scaled_velocity)
    amplitude_prediction = lm2.predict(scaled_amplitude)
    
    df['velocity_prediction'] = velocity_prediction
    df['amplitude_prediction'] = amplitude_prediction
    
    return X_train_scaled, X_test_scaled, y_train, y_test,df

def Support_Vector_Regression(df):
        
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn.metrics import classification_report,confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    X_org = df.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end', 'Time at half'],axis=1)
    
    #y = df['amplitude']
    y = df[['Peak Velocity','Time at half','Peak Amplitude','Mean Velocity']]
    
    #y.loc['Peak Amplitude'] /= 2
    
    # Split the data into train and test randomly with the test size = 30%, stratify data to split classification evenly
    X_train, X_test, y_train, y_test = train_test_split(X_org, y, test_size=0.30, random_state=42) #random_state=42
    
    #features_int = ['biceps_max'] #'curr_acc' biceps_max','curr_triceps_change2'] #'curr_acc'
    X_train_peak_vel = X_train #[features_int]
    X_test_peak_vel = X_test #[features_int]
    
    y_train_velocity = y_train.iloc[:,0]
    y_train_time = y_train.iloc[:,1]
    y_train_amplitude = y_train.iloc[:,2]
    y_train_mean_vel = y_train.iloc[:,3]
    
    y_test_velocity = y_test.iloc[:,0]
    y_test_time = y_test.iloc[:,1]
    y_test_amplitude = y_test.iloc[:,2]
    y_test_mean_vel = y_test.iloc[:,3]
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train_peak_vel)
    X_test_scaled = min_max_scaler.transform(X_test_peak_vel)
    
    clf = SVR(kernel = 'linear',C= 1, gamma = 1, epsilon=0.01,degree=2) # 10,100
    clf.fit(X_train_scaled, y_train_velocity)
    
    # Generate predictions for test set
    velocity_predictions = clf.predict(X_test_scaled)
    df_velocity_predictions = pd.DataFrame({'y_test':y_test_velocity, 'predictions':velocity_predictions})
        
    # Show cross-val results
    #print(confusion_matrix(y_test,predictions))
    #print(classification_report(y_test,predictions))
    from sklearn.metrics import r2_score
    
    print('Peak Velocity')
    print('MAE:', metrics.mean_absolute_error(y_test_velocity, velocity_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_velocity, velocity_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_velocity, velocity_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_velocity, velocity_predictions)/np.mean(y_test_velocity))
    print('Score:', clf.score(X_test_scaled,velocity_predictions),'\n')
    print('R2 Score:', r2_score(y_test_velocity, velocity_predictions),'\n')
    
    #print(classification_report(y_test_velocity, velocity_predictions))
    
    # print('Coefficients: ')
    # pd.set_option('display.max_columns', df.shape[1]+1)
    # coeffs = np.abs(clf.coef_)/np.sum(np.abs(clf.coef_))*100
    # df = pd.DataFrame(data = coeffs, columns = X_train_peak_vel.columns.tolist())
    # df = df.loc[0].abs().sort_values()
    # print(df,'\n')
    
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_velocity_predictions)
    plt.plot([0,200],[0,200],'r',lw=1)
    plt.title('Peak Velocity Prediction')
    
    #plt.scatter(y_test,predictions)
    plt.show()
    
    #%%%%%%%%%%% MEAN VEL %%%%%%%%%%%%%%
    from sklearn.model_selection import GridSearchCV 
    
    #features_int = ['biceps_max','biceps_sum','curr_acc','curr_velocity','triceps_min', 'curr_triceps_change2', 'pecs_sum','segment_duration'] #'triceps_max','pecs_sum','curr_triceps_change2'\
                   # ,'biceps_sum', 'deltoid_sum'] #'curr_acc'
    
    # print('Features used: ',features_int)
    # X_train_mean_vel = X_train[features_int]
    # X_test_mean_vel = X_test[features_int]
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler4 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler4.fit_transform(X_train)
    X_test_scaled = min_max_scaler4.transform(X_test)
    
    parameters = {'kernel': ['rbf'], 'C':[1,10,100,1000],'gamma': [10,1,1e2],'epsilon':[0.01,0.1,1,10]}

    svr2 = SVR() #kernel = 'poly',C=10,gamma=1,epsilon=0.01,degree=3) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    
    #clf4 = SVR('rbf',C= 100, gamma = 1, epsilon=0.1,degree=2)
    clf4 = GridSearchCV(svr2, parameters)
    
    clf4.fit(X_train_scaled,y_train_mean_vel)
    
    print(clf4.best_params_)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    mean_vel_predictions = clf4.predict(X_test_scaled)
    df_mean_vel_predictions = pd.DataFrame({'y_test':y_test_mean_vel, 'predictions':mean_vel_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_mean_vel_predictions)
    plt.title('Mean Velocity Prediction')
    #plt.plot([5,50],[5,50],'r',lw=1)
    a = min(y_test_mean_vel)
    b = max(y_test_mean_vel)
    
    plt.plot([a,b],[a,b],'r',lw=1)
    plt.legend(['Predicted Correlation','Real Correlation'])
    
    plt.xlabel('True Mean Velocity (degrees/s)')
    plt.ylabel('Predicted Mean Velocity (degrees/s)')
    plt.savefig("mean_vel_40%.svg",format='svg')
    plt.show()
    
    print('Mean Velocity')
    print('MAE:', metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions)/np.mean(y_test_mean_vel))
    print('Coefficients: ')
    print('Score:', clf4.score(X_test_scaled,mean_vel_predictions),'\n')
    print('R2 Score:', r2_score(y_test_mean_vel, mean_vel_predictions),'\n')
    
    #%%############# DURATION ###################
    
    
    # features_int2 = ['biceps_max', 'biceps_sum','biceps_min','curr_acc','curr_velocity','triceps_min', 'pecs_sum','segment_duration']# 'triceps_max','pecs_sum','curr_triceps_change2'\
    #                # ,'biceps_sum', 'deltoid_sum'] #'curr_acc'
    
    # print('Features used: ',features_int2)
    # X_train_dur = X_train[features_int2]
    # X_test_dur = X_test[features_int2]
    
    # X_train_dur['Predicted Mean Velocity'] = clf4.predict(X_train_scaled)
    # X_test_dur['Predicted Mean Velocity'] = clf4.predict(X_test_scaled)
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    # min_max_scaler2 = preprocessing.MinMaxScaler()
    # X_train_scaled = min_max_scaler2.fit_transform(X_train)
    # X_test_scaled = min_max_scaler2.transform(X_test)
    
    
    parameters = {'kernel': ['rbf'], 'C':[0.01,1,10,100],'gamma': [0.01,0.1,1,10],'epsilon':[0.01,0.1,1]}

    svr3 = SVR()   #clf2 = SVR(kernel = 'rbf',C=100,gamma=1,epsilon=0.1,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    
    clf2 = GridSearchCV(svr3, parameters)
    
    clf2.fit(X_train_scaled,y_train_time)
    
    print(clf2.best_params_)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    time_predictions = clf2.predict(X_test_scaled)
    df_time_predictions = pd.DataFrame({'y_test':y_test_time, 'predictions':time_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_time_predictions)
    plt.title('Duration Prediction')
    
    plt.plot([0.56,2.08],[0.56,2.08],'r',lw=1)
    plt.legend(['Predicted Correlation','Real Correlation'])
    
    plt.xlabel('True Trajectory Duration (s)')
    plt.ylabel('Predicted Trajectory Duration (s)')
    plt.savefig("duration_40%.svg",format='svg')
    plt.show()
    
    print('Duration')
    print('MAE:', metrics.mean_absolute_error(y_test_time, time_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_time, time_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_time, time_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_time, time_predictions)/np.mean(y_test_time))
    print('Score:', clf2.score(X_test_scaled,time_predictions),'\n')
    print('R2 Score:', r2_score(y_test_time, time_predictions),'\n')
    
    # print('Coefficients: ')
    # #pd.set_option('display.max_columns', df.shape[1]+1)
    # coeffs2 = np.abs(clf2.coef_)/np.sum(np.abs(clf2.coef_))*100
    # df = pd.DataFrame(data = coeffs2, columns = X_train.columns.tolist())
    # df = df.loc[0].abs().sort_values()
    # print(df,'\n')
    
    #%%%%%%%%%%% AMPLTIUDE %%%%%%%%%%%%%%
    #     # Scale the train data to range [0 1] and scale the test data according to the train data
    
    # #features_int = ['curr_acc']
    
    # X_train_amplitude = X_train.copy()
    # X_test_amplitude = X_test.copy()
    
    # #scaled_X_train = min_max_scaler4.fit_transform(X_train)
    # #scaled_X_test = min_max_scaler4.fit_transform(X_test)
    
    # mean_velocity_predictions_train = clf4.predict(X_train_scaled)
    # mean_velocity_predictions_test = clf4.predict(X_test_scaled)
    # peak_velocity_predictions_train = clf.predict(X_train_scaled)
    # peak_velocity_predictions_test = clf.predict(X_test_scaled)
    
    # X_train_amplitude_new = pd.DataFrame()
    # X_test_amplitude_new = pd.DataFrame()
    
    # X_train_amplitude['predicted mean velocity'] = mean_velocity_predictions_train
    # X_test_amplitude['predicted mean velocity'] = mean_velocity_predictions_test
    # X_train_amplitude['predicted peak velocity'] = peak_velocity_predictions_train
    # X_test_amplitude['predicted peak velocity'] = peak_velocity_predictions_test
    
    # min_max_scaler3 = preprocessing.MinMaxScaler()
    # X_train_scaled = min_max_scaler3.fit_transform(X_train_amplitude)
    # X_test_scaled = min_max_scaler3.transform(X_test_amplitude)
    
    # clf3 = SVR(kernel = 'rbf',C=10,gamma=0.1,epsilon=0.01,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    # clf3.fit(X_train_scaled,y_train_amplitude)
    
    # # The coefficients
    # #print('Coefficients: \n', lm.coef_)
    
    # amplitude_predictions = clf3.predict(X_test_scaled)
    # df_amplitude_predictions = pd.DataFrame({'y_test':y_test_amplitude, 'predictions':amplitude_predictions})
    
    # #plt.scatter(y_test,predictions)
    # sns.lmplot(x= 'y_test',y = 'predictions', data = df_amplitude_predictions)
    # plt.plot([20,100],[20,100],'r',lw=1)
    # plt.title('Amplitude Prediction')
    
    # plt.xlabel('Y Test')
    # plt.ylabel('Predicted Y')
    # plt.show()
    
    # print('Amplitude')
    # print('MAE:', metrics.mean_absolute_error(y_test_amplitude, amplitude_predictions))
    # print('MSE:', metrics.mean_squared_error(y_test_amplitude, amplitude_predictions))
    # print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_amplitude, amplitude_predictions)))
    # print('% Error: ', 100*metrics.mean_absolute_error(y_test_amplitude, amplitude_predictions)/np.mean(y_test_amplitude))
    # print('Score:', clf3.score(X_test_scaled,amplitude_predictions),'\n')
    # # print('Coefficients: ')
    
    # # #pd.set_option('display.max_columns', df.shape[1]+1)
    # # coeffs3 = np.abs(clf3.coef_)/np.sum(np.abs(clf3.coef_))*100
    # # df = pd.DataFrame(data = coeffs3, columns = X_train_amplitude.columns.tolist())
    # # df = df.loc[0].abs().sort_values()
    # # print(df,'\n')
    
    # # #pd.set_option('display.max_columns', df.shape[1]+1)
    # # coeffs4 = np.abs(clf4.coef_)/np.sum(np.abs(clf4.coef_))*100
    # # df = pd.DataFrame(data = coeffs4, columns = X_train.columns.tolist())
    # # df = df.loc[0].abs().sort_values()
    # # print(df,'\n')
    
    X_test2 = X_test.copy() 
    X_test2['peak_velocity_predictions'] = velocity_predictions
    X_test2['duration_predictions'] = time_predictions
    # X_test2['amplitude_predictions'] = amplitude_predictions
    X_test2['mean_vel_predictions'] = mean_vel_predictions
    
    return X_train, X_train_scaled, y_train, X_test2,y_test, clf4, clf2
    
# def predict_values(dataset,clf,clf2,min_max_scaler, min_max_scaler2):
    
#     dataset2 = dataset.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end'],axis=1)
    
#     scaled_velocity = min_max_scaler.transform(dataset2)
#     scaled_amplitude = min_max_scaler2.transform(dataset2)
    
#     velocity_prediction = clf.predict(scaled_velocity)
#     amplitude_prediction = clf2.predict(scaled_amplitude)
    
#     df_predictions = dataset.copy() 
    
#     df_predictions['velocity_prediction'] = velocity_prediction
#     #df_predictions['amplitude_prediction'] = amplitude_prediction
#     df_predictions['Time at end prediction'] = amplitude_prediction

#     return df_predictions

# def evaluate(df_predictions_SVM):
    
#     from sklearn import metrics
#     import seaborn as sns
#     import numpy as np
#     import matplotlib.pyplot as plt
        
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,2,1)
#     plt.scatter(df_predictions_SVM['Time at end'],df_predictions_SVM['Mean Velocity'])
#     plt.scatter(df_predictions_SVM['Time at end prediction'],df_predictions_SVM['velocity_prediction'])
#     plt.xlabel('Total Trajectory Duration')
#     plt.ylabel('Mean Trajectory Velocity')
#     plt.legend(['Real','predicted'])
    
#     time_at_end_err = 100*(df_predictions_SVM['Time at end prediction']-df_predictions_SVM['Time at end'])/df_predictions_SVM['Time at end']
#     mean_velocity_err = 100*(df_predictions_SVM['velocity_prediction']-df_predictions_SVM['Mean Velocity'])/df_predictions_SVM['Mean Velocity']
    
#     plt.subplot(1,2,2)
#     plt.scatter(time_at_end_err, mean_velocity_err)
#     plt.xlabel('Total Trajectory Duration % Error')
#     plt.ylabel('Mean Trajectory Velocity % Error')
    
#     plt.show()
    
#     print('Duration:\n')
#     print('MAE:', metrics.mean_absolute_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
#     print('MSE:', metrics.mean_squared_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
#     print('RMSE:', np.sqrt(metrics.mean_squared_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction'])))
    
#     #print(clf.score(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
#     #print(clf.coef_)
                
#     sns.lmplot(x= 'Time at end',y = 'Time at end prediction', data = df_predictions_SVM)
    
#     print('Mean Velocity:\n')
    
#     print('MAE:', metrics.mean_absolute_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction']))
#     print('MSE:', metrics.mean_squared_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction']))
#     print('RMSE:', np.sqrt(metrics.mean_squared_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction'])))
    
#     #print(clf.score(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
#     #print(clf.coef_)
                
#     sns.lmplot(x= 'Mean Velocity',y = 'velocity_prediction', data = df_predictions_SVM)
#     plt.show()
    
def mjtg(current, setpoint, frequency, move_time):
    trajectory = []
    trajectory_derivative = []
    timefreq = int(move_time * frequency)

    for time in range(1, timefreq):
        trajectory.append(
            current + (setpoint - current) *
            (10.0 * (time/timefreq)**3
             - 15.0 * (time/timefreq)**4
             + 6.0 * (time/timefreq)**5))

        trajectory_derivative.append(
            frequency * (1.0/timefreq) * (setpoint - current) *
            (30.0 * (time/timefreq)**2.0
             - 60.0 * (time/timefreq)**3.0
             + 30.0 * (time/timefreq)**4.0))

    return trajectory, trajectory_derivative

def plot_predicted_traj(df_predictions_SVM,trajectory):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    for segment_num in range(30,31):
    
        current = trajectory[segment_num].iloc[0,-3]
        #setpoint = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-3]*df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]
        setpoint = df_predictions_SVM.iloc[segment_num,-1]*df_predictions_SVM.iloc[segment_num,-2]*2
        
        frequency = 100
        time = df_predictions_SVM.iloc[segment_num,-2]*2#(df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]-current)/df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-2]
        
        traj1, traj_vel = mjtg(current, setpoint, frequency, time)
        
        traj = np.array(traj1)
        #time_ = np.where(traj < 5)[-1][0]
        
        end_segment_point = trajectory[segment_num].iloc[0,-3]+32.0
        
        traj = traj[traj>end_segment_point]
    
        # Create plot.
        #xaxis = [i / frequency for i in range(1, int(time * frequency))]
        
        xaxis = np.linspace(0.0,6.0,601)
        
        index_ = trajectory[segment_num].loc[(trajectory[segment_num]['angular position'] > end_segment_point)].index[0]- trajectory[segment_num].index[0]
        
    #    if len(traj) > len(trajectory[segment_num]):
    #        traj = traj[0:len(trajectory[segment_num])]
    #    else:
    #        trajectory[segment_num] = trajectory[segment_num].iloc[0:len(traj)]
        
        plt.figure(1,figsize=(5,5))
        plt.plot(xaxis[index_:len(traj)+index_], traj)
        plt.plot(xaxis[0:len(trajectory[segment_num].iloc[:,-3])], trajectory[segment_num].iloc[:,-3])
        
        #plt.plot(xaxis, traj_vel)
        plt.title("Minimum jerk trajectory")
        plt.xlabel("Time [s]")
        plt.ylim(0,100)
        plt.ylabel("Angle [deg]")
        plt.legend(['Predicted', 'Real'])
        plt.savefig('predicted_and_real_traj_40%.svg', format='svg')
    #    plt.figure(2)
    #    plt.plot(trajectory[segment_num]['time'],trajectory[segment_num]['imu'])
    #    
        plt.show()


def normalise_MVC(emg_MVC, dataset):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    """
    Function to normalise EMG signals to MVC
    
    """
    
    # plt.subplot(2,2,1)
    # plt.plot(emg_MVC['Time'], emg_MVC['Biceps'])
    # plt.title('biceps')
    
    # plt.subplot(2,2,2)
    # plt.plot(emg_MVC['Time'], emg_MVC['Triceps'])
    # plt.title('triceps')
    
    # plt.subplot(2,2,3)
    # plt.plot(emg_MVC['Time'], emg_MVC['Deltoid'])
    # plt.title('Deltoid')
    
    # plt.subplot(2,2,4)
    # plt.plot(emg_MVC['Time'], emg_MVC['Pecs'])
    # plt.title('Pecs')
    
    # plt.show()
    
    
    # filter the MVC signal
    biceps_env = filteremg(emg_MVC['Time'], emg_MVC['Biceps'], graphs=0)
    triceps_env = filteremg(emg_MVC['Time'], emg_MVC['Triceps'],  graphs=0)
    deltoid_env = filteremg(emg_MVC['Time'], emg_MVC['Deltoid'],  graphs=0)
    pecs_env = filteremg(emg_MVC['Time'], emg_MVC['Pecs'], graphs=0)
    
    
    # find the maximum values (search for the biceps maximum in the first half of the signal )
    a = int(0.5*len(biceps_env))
    biceps_MVC = np.max(biceps_env[0:a])
    triceps_MVC = np.max(triceps_env)
    deltoid_MVC  = np.max(deltoid_env)
    pecs_MVC = np.max(pecs_env)
    
    # normalise the signals based on the MVC
    dataset['biceps'] = dataset['biceps']/biceps_MVC
    dataset['biceps'] = dataset['biceps']/triceps_MVC
    dataset['biceps'] = dataset['biceps']/deltoid_MVC
    dataset['biceps'] = dataset['biceps']/pecs_MVC
    
    return dataset

#%%

def ANN(df):
    
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn.metrics import classification_report,confusion_matrix, mean_absolute_error, mean_squared_error, explained_variance_score
    from sklearn.preprocessing import MinMaxScaler
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.optimizers import Adam
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sys
    
    # Define X and y
    #X = df.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end', 'Time at half'],axis=1)
    features_int = ['biceps_max','biceps_sum','curr_acc','curr_velocity', 'segment_duration','triceps_min', 'curr_triceps_change2', 'pecs_sum']# 'triceps_max','pecs_sum','curr_triceps_change2'\
        
    X = df[features_int]     
    y = df[['Time at half','Mean Velocity']]
    
    # Split the data into train and test randomly with the test size = 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41) 
    
    # Create scalar
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Split y 
    y_test_vel = y_test.iloc[:,1]
    y_test_time = y_test.iloc[:,0]
    
    # Create ANN Model 
    model = Sequential()
    
    model.add(Dense(8,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4,activation='relu'))

    model.add(Dense(1))
    
    model.compile(optimizer='adam',loss='mse')
    
    model.fit(x= X_train, y= y_train.values, validation_data = (X_test,y_test_vel.values),batch_size=50, epochs = 300)
    
    losses = pd.DataFrame(model.history.history)
    
    losses.plot()
    
    ## Model Evaluation
    predictions = model.predict(X_test)
    
    print(mean_absolute_error(y_test_vel,predictions))
    print(mean_squared_error(y_test_vel,predictions))
    print((explained_variance_score(y_test_vel,predictions)))
    print(df['Mean Velocity'].mean())
    
    plt.figure(2)
    # Our predictions
    plt.scatter(y_test_vel,predictions)
    
    # Perfect predictions
    plt.plot(y_test_vel, y_test_vel,'r')
    
    return 
    
    # Outlier detection 

def detect_outliers(df,n,features):
    
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers 
    
#%% Regression models
def baseline_regression_models(X_train, y_train, mean_val, kfolds=0, n_jobs=1, combined = 0):
    
    classifiers = []
    classifiers.append(SVR())
    classifiers.append(DecisionTreeRegressor(random_state=42))
    classifiers.append(AdaBoostRegressor(DecisionTreeRegressor(random_state=42),random_state=42,learning_rate=0.1))
    classifiers.append(RandomForestRegressor(random_state=42))
    classifiers.append(ExtraTreesRegressor(random_state=42))
    classifiers.append(GradientBoostingRegressor(random_state=42))
    classifiers.append(MLPRegressor(random_state=42))
    classifiers.append(ElasticNet(random_state=42))
    classifiers.append(Lasso(random_state=42))
    classifiers.append(Ridge(random_state=42))
    
    if kfolds ==0:
        kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_results = []
    for classifier in classifiers :
        cv_results.append(-cross_val_score(classifier, X_train, y_train, scoring = "neg_mean_absolute_error", cv = kfolds, n_jobs=n_jobs))
    
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean()*100/mean_val)
        cv_std.append(cv_result.std()*100/mean_val)
    
    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVR","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","ElasticNet","Lasso","Ridge"]})
    
    plt.figure()
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res.sort_values('CrossValMeans'), palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean % Error")
    #g = g.set_title("Cross validation scores{}".format(mean_val))
    
    RFR = RandomForestRegressor(random_state=42)
    RFR.fit(X_train,y_train)
    #cross_val_error_ht.append(np.mean(-cross_val_score(baseline, X_train, y_train_mv, scoring = "neg_mean_absolute_error", cv = kfolds, n_jobs=n_jobs))*100/mean_mean_velocity)

    predictors = list(X_train)
    feat_imp = pd.Series(RFR.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp[:20].plot(kind='bar', title='Importance of Features')
    
    time_imp = 0
    imu_imp = 0
    stretch_imp = 0
    emg_imp = 0
    
    print(cv_res.sort_values('CrossValMeans').reset_index(drop=True))
    
    if combined ==1:
    
        for item in feat_imp.index:
            #print(item)
            
            x = feat_imp.loc[item]
            
            if 'time' in item:
                
                time_imp = time_imp + x
                
            if 'pos' in item or 'vel' in item or 'acc' in item:
                
                imu_imp = imu_imp + x
                
            if 'stretch' in item:
                
                stretch_imp = stretch_imp + x
            
            if 'bb' in item or 'tb' in item  or 'ad' in item or 'pm' in item:
                
                emg_imp = emg_imp + x
        
        sensor_imp_lst = [time_imp, imu_imp, stretch_imp, emg_imp]
    
        sensor_imp = pd.Series(sensor_imp_lst,['time','imu','stretch','emg' ])
    
    
        return cv_res.sort_values('CrossValMeans').reset_index(drop=True), feat_imp, sensor_imp
    
    return cv_res.sort_values('CrossValMeans').reset_index(drop=True), feat_imp
    
#%%


def combine_extracted_dataframes(extracted_features_test, trajectory_test, angle_cutoff):

    """
    Combines the extracted features dataframe from each trial into 1 dictionary containing 1 big dataframe for each window length
    Combines all the trajecories from different trials into one dictionary

    """    

    full_combined_features = {}
    
    for window in range(0,len(angle_cutoff)):
        full_combined_features[window] = extracted_features_test[0][window].copy()
    
        for j in range(1,6):
            
            full_combined_features[window] = full_combined_features[window].append(extracted_features_test[j][window], ignore_index=True, sort=False)
            
    full_combined_trajectories = {} #trajectory_test[0].copy()
    x=0
    
    for k in range(0,6):
        for t in range(0,len(trajectory_test[0])):
            
            full_combined_trajectories[x] = trajectory_test[k][t]
            x=x+1
            
    return full_combined_features, full_combined_trajectories

#%%
"""
Filter the features from the MJT R2 score

"""

def filter_trajectories_by_MJT_sim(full_combined_trajectories,full_combined_features, angle_cutoff,sim_score = 0.85):

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
        
        if r2score[h] < sim_score:
            del filtered_mj_traj[h]
            
            for i in range(0,len(angle_cutoff)): ###NUMBER OF DATASETS
            
                #dataset_slct = filtered_mj_feat[i]
                filtered_mj_feat[i].drop([h],inplace=True)
    
    for i in range(0,len(angle_cutoff)):
        filtered_mj_feat[i] = filtered_mj_feat[i].reset_index(drop=True)
                
    
    # RESET INDEX:
    filtered_mj_traj = {i: v for i, v in enumerate(filtered_mj_traj.values())}
    
    return filtered_mj_feat, filtered_mj_traj

def remove_outliers(filtered_mj_feat, filtered_mj_traj, outlier_threshold, angle_cutoff):

    outlier_rows_per_dataframe = {} # this can be used to select trajectories
    
    trajectories = {}
        
    for j in filtered_mj_traj.keys():
        filtered_mj_traj[j] = filtered_mj_traj[j].reset_index(drop=True)
        
    for i in range(0,len(angle_cutoff)):
        
        num_of_outliers_threshold= outlier_threshold*len(filtered_mj_feat[i].columns)
        
        Outliers_to_drop = detect_outliers(filtered_mj_feat[i], num_of_outliers_threshold ,list(filtered_mj_feat[i].columns.values))
        
        outlier_rows_per_dataframe[i] = Outliers_to_drop
        
        filtered_mj_feat[i].drop(outlier_rows_per_dataframe[i], axis = 0,inplace=True)
            
        filtered_mj_feat[i] = filtered_mj_feat[i].reset_index(drop=True)
        
        copy = filtered_mj_traj.copy()
        
        for j in outlier_rows_per_dataframe[i]:
            del copy[j]
            
        trajectories[i] = copy
        trajectories[i] = {i: v for i, v in enumerate(trajectories[i].values())}
    
    return filtered_mj_feat, trajectories, outlier_rows_per_dataframe





























    
    
    
    
    
    