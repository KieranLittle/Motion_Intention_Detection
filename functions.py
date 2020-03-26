#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:17:40 2020

@author: Kieran
"""

def read_file(filename, trial_num):
    
    import pandas as pd
    
    global amplitudes
    
    stretch_imu = pd.read_csv(filename+'/imu/Trial'+trial_num+'.csv', header=None)
    emg = pd.read_csv(filename+'/emg/Trial'+trial_num+'.csv', header=None)#'/emg/Trial4_emg.csv'
    amplitude = pd.read_csv(filename+'/Trajectory.csv', header=None)
    emg_MVC = pd.read_csv(filename+'/emg/MVC.csv', header=None)
    
    amplitude2 = amplitude[amplitude.iloc[:,1] != 0]
    amplitudes = []
    amplitudes.append(amplitude2.iloc[0,1])

    for i in range(len(amplitude2)-1):
        if (amplitude2.iloc[i,1] != amplitude2.iloc[i+1,1]):
            amplitudes.append(amplitude2.iloc[i+1,1])
    
    emg = emg.T
    emg.columns = ['Time','Biceps','Triceps','Deltoid','Pecs']
    
    emg_MVC = emg_MVC.T
    emg_MVC.columns = ['Time','Biceps','Triceps','Deltoid','Pecs']

    stretch_imu = stretch_imu.T
    stretch_imu.columns = ['time','imu','stretch']
    
    combined_dataset = stretch_imu
    combined_dataset['Biceps'] = emg['Biceps']
    combined_dataset['Triceps'] = emg['Triceps']
    combined_dataset['Deltoid'] = emg['Deltoid']
    combined_dataset['Pecs'] = emg['Pecs']
    
    return combined_dataset, amplitudes, emg_MVC

def resample(dataset):
    
    import scipy
    import numpy as np
    from scipy import signal
    import pandas as pd
    
    #a = dataset[dataset['time'] == 1200.0].index
    #dataset.drop(a , inplace=True)
    
    dataset = dataset.loc[:200000];
    
    resampled_stretch = scipy.signal.resample(dataset.iloc[:,2],120001)
    resampled_imu = scipy.signal.resample(dataset.iloc[:,1],120001)
    resampled_BB = scipy.signal.resample(dataset.iloc[:,3],120001)
    resampled_TB = scipy.signal.resample(dataset.iloc[:,4],120001)
    resampled_D = scipy.signal.resample(dataset.iloc[:,5],120001)
    resampled_P = scipy.signal.resample(dataset.iloc[:,6],120001)
    
    time = np.linspace(0,12000*0.1,120001)

    dataset = pd.DataFrame({'time': time, 'stretch': resampled_stretch, 'imu': resampled_imu, 'biceps': resampled_BB,\
                            'triceps': resampled_TB, 'deltoid': resampled_D, 'pecs': resampled_P})
    
    decimals = 2 
    dataset['time'] = dataset['time'].apply(lambda x: round(x, decimals))
    dataset['imu'] = dataset['imu'].apply(lambda x: round(x, decimals))
    
    return dataset
    

def filteremg(time, emg, low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=0):
    import scipy as sp
    import matplotlib.pyplot as plt
    
    
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band/(sfreq/2)
    low_band = low_band/(sfreq/2)
    
    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)    
    
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/sfreq
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)
    
    if graphs == 1:
    
        # plot graphs
        fig = plt.figure()
        plt.subplot(1, 4, 1)
        plt.subplot(1, 4, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
        plt.plot(time, emg)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-0.1, 0.1)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')
        
        plt.subplot(1, 4, 2)
        plt.subplot(1, 4, 2).set_title('Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band*sfreq)) + '-' + str(int(low_band*sfreq)) + 'Hz')
        plt.plot(time, emg_rectified)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-0.1, 0.1)
        plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
        plt.xlabel('Time (sec)')
    
        plt.subplot(1, 4, 3)
        plt.subplot(1, 4, 3).set_title('Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass*sfreq)) + ' Hz')
        plt.plot(time, emg_envelope)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-0.1, 0.1)
        plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
        plt.xlabel('Time (sec)')
        
        plt.subplot(1, 4, 4)
        plt.subplot(1, 4, 4).set_title('Focussed region')
        plt.plot(time[int(0.9*1000):int(1.0*1000)], emg_envelope[int(0.9*1000):int(1.0*1000)])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.xlim(0.9, 1.0)
        #plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
    
        fig_name = 'fig_' + str(int(low_pass*sfreq)) + '.png'
        fig.set_size_inches(w=11,h=7)
        fig.savefig(fig_name)
        
        plt.show()

 # show what different low pass filter cut-offs do
#    for i in [3, 10, 40]:
#            filteremg(time, emg, low_pass=i) #emg_correctmean
    return emg_envelope

def lowpass_filter(time, signal, order=2, low_pass=10, sfreq=1000):
    
    import scipy as sp
    import matplotlib.pyplot as plt
    
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/sfreq
    b, a = sp.signal.butter(1, low_pass, btype='lowpass')
    filtered_signal = sp.signal.filtfilt(b, a, signal)#imu_rectified)
    
    # plt.figure(1, figsize=(15,8))
    # plt.subplot(2,5,1)
    # plt.plot(time[0:10000],signal[0:10000])
    # plt.subplot(2,5,2)
    # plt.plot(time[0:10000],filtered_signal[0:10000])
    # plt.subplot(2,5,3)
    # plt.plot(time[0:1000],signal[0:1000])
    # plt.subplot(2,5,4)
    # plt.plot(time[0:1000],filtered_signal[0:1000])
    # plt.subplot(2,5,5)
    # plt.plot(time[0:1000],signal[0:1000])
    # plt.plot(time[0:1000],filtered_signal[0:1000],'r')
    # plt.show()
    
    return filtered_signal

def filter_smooth(old_dataset):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #N = 15
    #imu_smooth = np.convolve(old_dataset['imu'], np.ones((N,))/N, mode='valid')
    
    new_dataset = old_dataset.copy()
    
    new_dataset['angular position'] = lowpass_filter(new_dataset['time'],new_dataset['imu'],3,2,100)
    
    new_dataset.drop('imu',axis=1,inplace=True)
    
    vel = np.diff(new_dataset['angular position'],1)/0.01
    vel = np.append(vel,0)
    
    new_dataset['angular velocity'] = lowpass_filter(new_dataset['time'],vel,2,7,100)
    
    acc = np.diff(new_dataset['angular velocity'],1)/0.01
    acc = np.append(acc,0)
    
    new_dataset['angular acceleration'] = lowpass_filter(new_dataset['time'],acc,2,9,100)
    #new_dataset['angular acceleration'] = filteremg(new_dataset['time'],acc,low_pass=5, sfreq=100, high_band=20, low_band=450, graphs=1)
    
    new_dataset['stretch'] = lowpass_filter(new_dataset['time'],new_dataset['stretch'],2,4,100)
    
#    new_dataset.drop(new_dataset.index[0:N-1],inplace=True) 
#    new_dataset['imu']=imu_smooth
#    new_dataset.reset_index(drop=True, inplace = True)
#     
    biceps_filtered = filteremg(new_dataset['time'],new_dataset['biceps'])
    triceps_filtered = filteremg(new_dataset['time'],new_dataset['triceps'])
    deltoid_filtered = filteremg(new_dataset['time'],new_dataset['deltoid'])
    pecs_filtered = filteremg(new_dataset['time'],new_dataset['pecs'])
    
    new_dataset['biceps']=biceps_filtered
    new_dataset['triceps']=triceps_filtered
    new_dataset['deltoid']=deltoid_filtered
    new_dataset['pecs']=pecs_filtered

    return new_dataset

def segment_data(dataset,amplitudes):
    
    import numpy as np
    import pandas as pd
    
    count = 0
    #amplitude_var = np.zeros(len(dataset))
    amplitude_temp = []
    amplitude_change =[]
    half_amplitude_change = []
    
    start_angle = []
    traj_segment ={}
    trajectory = {}
    trajectory_temp = {}
    
    time_start = []
    time_half = []
    time_end = []
    
    peak_segment_velocity = []
    mean_segment_velocity = []
    segment_max_point = []
    segment_cut_off = []
    
    for j in range(0,len(dataset['time'])):
      if (dataset['time'][j] % 8.0 == 0.0) & (dataset['time'][j] != 0.0) & (dataset['time'][j] != 1200.0): #& (dataset['imu'][j] < 5):
          
          # Define initial trajectory
          trajectory_temp[count] = dataset.iloc[j:j+700,:] #400
          f = trajectory_temp[count].loc[((trajectory_temp[count].iloc[:,-3] > np.max(trajectory_temp[count].iloc[:,-3]-5)))] #& (trajectory_temp[count].iloc[:,-2]>0))] 
          end = f.index[0]- dataset.index[j]                               
          #trajectory[count] = trajectory_temp[count].iloc[0:f.index[0]]
                 
          # Define the end point of the trajectory:
          #amp_range = trajectory[count].loc[(trajectory[count]['angular position'] > max_amp-5)]
          
          max
          end_traj = trajectory_temp[count][(end-50):]
          
          #diff
          
          # try:
          #     limit = 0.01#+dataset.iloc[j,-2]
          #     amp_range = end_traj.loc[(end_traj.iloc[:,-2] < limit)] 
          #     end_amplitude_index = amp_range.index[0]+40
          # except:
          #     try:
          #         limit = 0.05#+dataset.iloc[j,-2]
          #         amp_range = end_traj.loc[(end_traj.iloc[:,-2] < limit)] 
          #         end_amplitude_index = amp_range.index[0]+40
          #     except:
 
          #         end_amplitude_index = end_traj.index[-1]
              
          amp_range = end_traj.loc[(end_traj.iloc[:,-2] < 0.01+min(abs(end_traj.iloc[:,-2])))] #0.99*max(end_traj.iloc[:,-3]))] 
          end_amplitude_index = amp_range.index[0]
              
          #amp_range = end_traj.loc[(end_traj.iloc[:,-2] < limit)] #+dataset.iloc[j,-2])]
          
          #amp_range = trajectory[count].loc[(trajectory[count].iloc[:,-2] > 0.02+dataset.iloc[j,-2])&(trajectory[count]['angular velocity']>0)&(trajectory[count]['angular acceleration']<0)]
          #end_amplitude_index = amp_range.index[0]
          
          # Define the start point of the trajectory
          #a = trajectory[count].loc[(trajectory[count]['angular acceleration']) > 0.001] #(10.0+dataset['angular position'][j]
          # p = (trajectory[count]['angular velocity']) > 0.01+dataset.iloc[j,-2]
          # z = p>0
          
          #a = trajectory[count].loc[(trajectory[count]['angular velocity'] > 0.01)&(trajectory[count]['angular velocity']>0)]  
          a = trajectory_temp[count].loc[(trajectory_temp[count].iloc[:,-2] > -0.01)]   #&(trajectory[count]['angular velocity']>0)]  
          
          start_amplitude_index_temp = a.index[0]
          
          # Segment trajectory:        
          trajectory_temp[count] = dataset.iloc[start_amplitude_index_temp:end_amplitude_index,:]#end_amplitude_index,:]
          
          mean_velocity_temp = np.mean(trajectory_temp[count]['angular velocity'])  
          start_amplitude_index = start_amplitude_index_temp#-int((mean_velocity_temp//2))
          end_amplitude_index = end_amplitude_index#+int((mean_velocity_temp//2))
          
          trajectory[count] = dataset.iloc[start_amplitude_index_temp:end_amplitude_index,:]
          
          # Define window:
          
          b = trajectory[count].loc[(trajectory[count].iloc[:,-3] < 0.5*(np.max(trajectory[count].iloc[:,-3])-trajectory[count].iloc[0,-3]))]      #+trajectory[count].iloc[0,-3])] : 0.25*(trajectory[count].iloc[-1,-3]-trajectory[count].iloc[-1,-3]
          
          traj_segment[count] = dataset.iloc[start_amplitude_index:b.index[-1],:] 
          
          # Measure variables that define the trajectory: 
          
          # Find the start and end angular position and the change
          end_amplitude =  trajectory[count].iloc[-1,-3]#trajectory[count].iloc[amp_range_index,-3]
          start_amplitude = trajectory[count].iloc[0,-3]
          amplitude_change_ = end_amplitude- start_amplitude
          amplitude_change.append(amplitude_change_)
          
          # Find the time at half the cycle
          d = trajectory[count].loc[((trajectory[count]['angular position'] - start_amplitude) < (amplitude_change_/2))]
          

          half_time_ = d.iloc[-1,0]- trajectory[count].iloc[0,0]
          amplitude_half = d.iloc[-1,-3] #- start_amplitude
          
          time_half.append(half_time_)
            
          # Find the amplitude at halfway
          #amplitude_half = d.iloc[-1,-3] - start_amplitude
          half_amplitude_change.append(amplitude_half)
          
          # Find peak and mean velocity
          peak_velocity = np.max(trajectory[count]['angular velocity'])   #np.max(np.diff(dataset[j:j+400]['imu'],1))/0.01
          mean_velocity = np.mean(trajectory[count]['angular velocity'])  
          peak_segment_velocity.append(peak_velocity)
          mean_segment_velocity.append(mean_velocity)
          
          segment_cut_off.append(start_amplitude_index)
          segment_max_point.append(end_amplitude_index)
          
          # Find the start position
          start_angle.append(start_amplitude)
          
          time_start_ = trajectory[count].iloc[0,0]
          time_end_ = trajectory[count].iloc[-1,0]-time_start_
          
          #time_start.append(time_start_)
          time_end.append(time_end_)
          
          count=count+1
          
    peak_segment_velocity = [ round(elem, 2) for elem in peak_segment_velocity ]
    time_end = [ round(elem, 2) for elem in time_end ]
    time_half = [ round(elem, 2) for elem in time_half ]

#          
    dict = {'amplitude_change': amplitude_change, 'peak_velocity': peak_segment_velocity, 'mean_velocity': mean_segment_velocity, \
            'time_half':time_half, 'time_end':time_end, 'segment_cut_off':segment_cut_off, 'segment_max_point':segment_max_point, 'start_angle':start_angle}
        
    df = pd.DataFrame(dict)
        
    return traj_segment, trajectory,df

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
    

def extract_features(segments, df):
    
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

  stretch_sum = []
  imu_sum = []
  biceps_sum = []
  triceps_sum = []
  deltoid_sum = []
  pecs_sum = []
  #amplitude =[]

  biceps_max = []
  triceps_max = []
  deltoid_max = []
  
  stretch_max = []
  imu_max = []
  velocity_max = []
  
  curr_velocity = []
  curr_acc = []
  curr_biceps_change = []
  curr_biceps_change2 = []
  curr_triceps_change = []
  curr_triceps_change2 = []
  curr_stretch_change = []
  curr_stretch_change2 = []

  biceps_diff = []
  triceps_diff = []
  deltoid_diff = []
  stretch_diff = []
  imu_diff = []
  
  stretch_gradient = []
  biceps_gradient = []
  triceps_gradient = []
  deltoid_gradient = []
  pecs_gradient = []
  
  biceps_min = []
  biceps_min_time = []
  
  triceps_min = []
  deltoid_min = []
  
  segment_duration = []

  for segment_number in range(0,len(segments)):
      
    ## Time
      segment_duration.append(segments[segment_number].iloc[-1,0] - segments[segment_number].iloc[0,0])
      
    ## Position  
      
      imu_sum.append(segments[segment_number]['angular position'].sum())
      imu_max.append(np.max(segments[segment_number]['angular position']))
      imu_diff.append(segments[segment_number].iloc[-1]['angular position'] - segments[segment_number].iloc[0]['angular position'])
      
    ## Velocity
      velocity_max.append(np.max(segments[segment_number]['angular velocity']))
      curr_velocity.append(np.mean(segments[segment_number].iloc[-30:,-2]))
      
    ## Acceleration
      curr_acc.append(np.mean(segments[segment_number].iloc[-30:,-1]))#[(len(segments[segment_number])-30):]))
      
    ## Stretch
      stretch_sum.append(segments[segment_number]['stretch'].sum())
      stretch_max.append(np.max(segments[segment_number].iloc[:-10,1]))
      curr_stretch_change.append(np.mean(np.diff(segments[segment_number]['stretch'],1)[(len(segments[segment_number])-30):]))
      curr_stretch_change2.append(np.mean(np.diff(segments[segment_number]['stretch'],2)[(len(segments[segment_number])-30):]))
      stretch_diff.append(segments[segment_number].iloc[-1]['stretch'] - segments[segment_number].iloc[0]['stretch'])
      stretch_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,1],1)/0.01))
      
    ## Biceps
      
      biceps_sum.append(segments[segment_number]['biceps'].sum())
      biceps_max.append(np.max(np.abs(segments[segment_number]['biceps'])))
      curr_biceps_change.append(np.mean(np.diff(segments[segment_number]['biceps'],1)[(len(segments[segment_number])-30):]))
      curr_biceps_change2.append(np.mean(np.diff(segments[segment_number]['biceps'],2)[(len(segments[segment_number])-30):]))
      biceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,3],1)/0.01))
      biceps_diff.append(segments[segment_number].iloc[-1]['biceps'] - segments[segment_number].iloc[0]['biceps'])
      biceps_min.append(np.min(segments[segment_number]['biceps']))
      biceps_min_time.append(segments[segment_number].loc[segments[segment_number].idxmin()['biceps']]['time']-segments[segment_number].iloc[0,0])
       
    ## Triceps
      triceps_sum.append(segments[segment_number]['triceps'].sum())
      triceps_max.append(np.max(np.abs(segments[segment_number]['triceps'])))
      curr_triceps_change.append(np.mean(np.diff(segments[segment_number]['triceps'],1)[(len(segments[segment_number])-30):]))
      curr_triceps_change2.append(np.mean(np.diff(segments[segment_number]['triceps'],2)[(len(segments[segment_number])-30):]))
      triceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,4],1)/0.01))
      triceps_diff.append(segments[segment_number].iloc[-1]['triceps'] - segments[segment_number].iloc[0]['triceps'])
      triceps_min.append(np.min(segments[segment_number]['triceps']))
    
    ## Deltoid
      deltoid_sum.append(segments[segment_number]['deltoid'].sum())
      deltoid_max.append(np.max(np.abs(segments[segment_number]['deltoid'])))
      deltoid_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,5],1)/0.01))
      deltoid_diff.append(segments[segment_number].iloc[-1]['deltoid'] - segments[segment_number].iloc[0]['deltoid'])
      deltoid_min.append(np.min(segments[segment_number]['deltoid']))
      
    ## Pecs
      pecs_sum.append(segments[segment_number]['pecs'].sum())
      pecs_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,6],1)/0.01))
     
  dict = {'curr_triceps_change2': curr_triceps_change2, 'curr_biceps_change2': curr_biceps_change2,\
          'triceps_min':triceps_min, 'triceps_max':triceps_max,'pecs_sum':pecs_sum,\
          'biceps_sum':biceps_sum,'biceps_max':biceps_max, 'biceps_min':biceps_min, \
          'curr_velocity':curr_velocity, 'curr_acc':curr_acc, \
          'segment_duration': segment_duration,'deltoid_sum':deltoid_sum,\
          'stretch_max':stretch_max}
   
      # 'stretch_gradient':stretch_gradient,
      
      #'biceps_min':biceps_min,'biceps_sum':biceps_sum, 'curr_biceps_change2': curr_biceps_change2, 'curr_triceps_change2': curr_triceps_change2} 
      #'window_max_velocity':velocity_max, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc, 'biceps_diff':biceps_diff, 'imu_diff':imu_diff, 'biceps_min':biceps_min, 'biceps_min_time':biceps_min_time } #{'biceps_sum':biceps_sum, 'biceps_max':biceps_max, 'biceps_gradient':biceps_gradient, 'triceps_gradient':triceps_gradient,'stretch_gradient':stretch_gradient, 'imu_diff':imu_diff, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc }
      #'deltoid_gradient':deltoid_gradient, 'pecs_gradient':pecs_gradient, 'imu_max':imu_max, 'stretch_gradient':stretch_gradient,'stretch_max':stretch_max} # 'stretch_gradient':stretch_gradient, 'window_max_velocity':velocity_max'stretch_gradient':stretch_gradient 'imu_diff':imu_diff 'window_max_velocity':velocity_max}  #'stretch_diff':stretch_gradient #stretch_diff':stretch_diff} #'velocity_max':velocity_max, 'biceps_max':biceps_max, 'triceps_diff':triceps_diff, 'deltoid_diff':deltoid_diff 'velocity_max':velocity_max 'stretch_diff':stretch_diff,'imu_diff':imu_diff,  'triceps_sum':triceps_sum,'deltoid_sum':deltoid_sum,'pecs_sum':pecs_sum,'amplitude':amplitude}
      
  df2 = pd.DataFrame(dict) 
      
    #df2['End Window Time'] = time_3degrees
       
    #df['peak_segment_velocity'] = peak_segment_velocity
  df2['Peak Amplitude'] = df['amplitude_change']
  df2['Peak Velocity'] = df['peak_velocity'] 
  df2['Mean Velocity'] = df['mean_velocity']
    
    #df['Time at half traj'] = time_half
  df2['Time at end'] = df['time_end']
  df2['Time at half'] = df['time_half']
      
    #df2 = df2[(np.abs(stats.zscore(df2)) < 2).all(axis=1)]
    
  #sns.pairplot(df2)
  #plt.show()
      
  return df2

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
    
    print('Peak Velocity')
    print('MAE:', metrics.mean_absolute_error(y_test_velocity, velocity_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_velocity, velocity_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_velocity, velocity_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_velocity, velocity_predictions)/np.mean(y_test_velocity))
    print('Score:', clf.score(X_test_scaled,velocity_predictions),'\n')
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
    
    features_int = ['biceps_max','biceps_sum','curr_acc','curr_velocity', 'segment_duration','triceps_min', 'curr_triceps_change2', 'pecs_sum']# 'triceps_max','pecs_sum','curr_triceps_change2'\
                   # ,'biceps_sum', 'deltoid_sum'] #'curr_acc'
    
    print('Features used: ',features_int)
    X_train_mean_vel = X_train[features_int]
    X_test_mean_vel = X_test[features_int]
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler4 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler4.fit_transform(X_train_mean_vel)
    X_test_scaled = min_max_scaler4.transform(X_test_mean_vel)
    
    parameters = {'kernel': ['rbf'], 'C':[1,10,100,1000],'gamma': [10,1,1e2],'epsilon':[0.01,0.1,1,10]}

    svr = SVR() #kernel = 'poly',C=10,gamma=1,epsilon=0.01,degree=3) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    
    #clf4 = SVR('rbf',C= 100, gamma = 1, epsilon=0.1,degree=2)
    clf4 = GridSearchCV(svr, parameters)
    
    clf4.fit(X_train_scaled,y_train_mean_vel)
    
    print(clf4.best_params_)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    mean_vel_predictions = clf4.predict(X_test_scaled)
    df_mean_vel_predictions = pd.DataFrame({'y_test':y_test_mean_vel, 'predictions':mean_vel_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_mean_vel_predictions)
    plt.title('Mean Velocity Prediction')
    plt.plot([5,50],[5,50],'r',lw=1)
    
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('Mean Velocity')
    print('MAE:', metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions)/np.mean(y_test_mean_vel))
    print('Coefficients: ')
    print('Score:', clf4.score(X_test_scaled,mean_vel_predictions),'\n')
    
    
    
    #%%############# DURATION ###################
    
    
    features_int2 = ['biceps_max', 'biceps_sum','biceps_min','curr_acc','curr_velocity', 'segment_duration','triceps_min', 'pecs_sum']# 'triceps_max','pecs_sum','curr_triceps_change2'\
                   # ,'biceps_sum', 'deltoid_sum'] #'curr_acc'
    
    print('Features used: ',features_int2)
    X_train_dur = X_train[features_int2]
    X_test_dur = X_test[features_int2]
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler2 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler2.fit_transform(X_train_dur)
    X_test_scaled = min_max_scaler2.transform(X_test_dur)
    
    parameters = {'kernel': ['rbf'], 'C':[0.01,1,10,100],'gamma': [0.01,0.1,1,10],'epsilon':[0.01,0.1,1]}

    svr = SVR()   #clf2 = SVR(kernel = 'rbf',C=100,gamma=1,epsilon=0.1,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    
    clf2 = GridSearchCV(svr, parameters)
    
    clf2.fit(X_train_scaled,y_train_time)
      
    print(clf2.best_params_)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    time_predictions = clf2.predict(X_test_scaled)
    df_time_predictions = pd.DataFrame({'y_test':y_test_time, 'predictions':time_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_time_predictions)
    plt.title('Duration Prediction')
    plt.plot([0.5,2.5],[0.5,2.5],'r',lw=1)
    
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('Duration')
    print('MAE:', metrics.mean_absolute_error(y_test_time, time_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_time, time_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_time, time_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_time, time_predictions)/np.mean(y_test_time))
    print('Score:', clf2.score(X_test_scaled,time_predictions),'\n')
    
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
    
    return X_test2,y_test
    
def predict_values(dataset,clf,clf2,min_max_scaler, min_max_scaler2):
    
    dataset2 = dataset.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end'],axis=1)
    
    scaled_velocity = min_max_scaler.transform(dataset2)
    scaled_amplitude = min_max_scaler2.transform(dataset2)
    
    velocity_prediction = clf.predict(scaled_velocity)
    amplitude_prediction = clf2.predict(scaled_amplitude)
    
    df_predictions = dataset.copy() 
    
    df_predictions['velocity_prediction'] = velocity_prediction
    #df_predictions['amplitude_prediction'] = amplitude_prediction
    df_predictions['Time at end prediction'] = amplitude_prediction

    return df_predictions

def evaluate(df_predictions_SVM):
    
    from sklearn import metrics
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
        
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(df_predictions_SVM['Time at end'],df_predictions_SVM['Mean Velocity'])
    plt.scatter(df_predictions_SVM['Time at end prediction'],df_predictions_SVM['velocity_prediction'])
    plt.xlabel('Total Trajectory Duration')
    plt.ylabel('Mean Trajectory Velocity')
    plt.legend(['Real','predicted'])
    
    time_at_end_err = 100*(df_predictions_SVM['Time at end prediction']-df_predictions_SVM['Time at end'])/df_predictions_SVM['Time at end']
    mean_velocity_err = 100*(df_predictions_SVM['velocity_prediction']-df_predictions_SVM['Mean Velocity'])/df_predictions_SVM['Mean Velocity']
    
    plt.subplot(1,2,2)
    plt.scatter(time_at_end_err, mean_velocity_err)
    plt.xlabel('Total Trajectory Duration % Error')
    plt.ylabel('Mean Trajectory Velocity % Error')
    
    plt.show()
    
    print('Duration:\n')
    print('MAE:', metrics.mean_absolute_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
    print('MSE:', metrics.mean_squared_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction'])))
    
    #print(clf.score(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
    #print(clf.coef_)
                
    sns.lmplot(x= 'Time at end',y = 'Time at end prediction', data = df_predictions_SVM)
    
    print('Mean Velocity:\n')
    
    print('MAE:', metrics.mean_absolute_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction']))
    print('MSE:', metrics.mean_squared_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction']))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(df_predictions_SVM['Mean Velocity'], df_predictions_SVM['velocity_prediction'])))
    
    #print(clf.score(df_predictions_SVM['Time at end'], df_predictions_SVM['Time at end prediction']))
    #print(clf.coef_)
                
    sns.lmplot(x= 'Mean Velocity',y = 'velocity_prediction', data = df_predictions_SVM)
    plt.show()
    
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
    
    
    for segment_num in range(40,80):
    
        current = trajectory[segment_num].iloc[0,2]
        setpoint = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-3]*df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]
        frequency = 100
        time = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-3]#(df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]-current)/df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-2]
        
        traj1, traj_vel = mjtg(current, setpoint, frequency, time)
        
        traj = np.array(traj1)
        #time_ = np.where(traj < 5)[-1][0]
        
        end_segment_point = trajectory[segment_num].iloc[0,2]+5.0
        
        traj = traj[traj>end_segment_point]
    
        # Create plot.
        #xaxis = [i / frequency for i in range(1, int(time * frequency))]
        
        xaxis = np.linspace(0.0,6.0,601)
        
        index_ = trajectory[segment_num].loc[(trajectory[segment_num]['imu'] > end_segment_point)].index[0]- trajectory[segment_num].index[0]
        
    #    if len(traj) > len(trajectory[segment_num]):
    #        traj = traj[0:len(trajectory[segment_num])]
    #    else:
    #        trajectory[segment_num] = trajectory[segment_num].iloc[0:len(traj)]
        
        plt.figure(1,figsize=(5,5))
        plt.plot(xaxis[index_:len(traj)+index_], traj)
        plt.plot(xaxis[0:len(trajectory[segment_num].iloc[:,2])], trajectory[segment_num].iloc[:,2])
        
        #plt.plot(xaxis, traj_vel)
        plt.title("Minimum jerk trajectory")
        plt.xlabel("Time [s]")
        plt.ylim(0,100)
        plt.ylabel("Angle [deg]")
        plt.legend(['Predicted', 'Real'])
        
    #    plt.figure(2)
    #    plt.plot(trajectory[segment_num]['time'],trajectory[segment_num]['imu'])
    #    
        plt.show()


def normalise_MVC(emg_MVC, dataset):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    
    plt.show()
    
    biceps_env = filteremg(emg_MVC['Time'], emg_MVC['Biceps'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=0)
    triceps_env = filteremg(emg_MVC['Time'], emg_MVC['Triceps'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=0)
    deltoid_env = filteremg(emg_MVC['Time'], emg_MVC['Deltoid'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=0)
    pecs_env = filteremg(emg_MVC['Time'], emg_MVC['Pecs'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=0)
    
    plt.show()
    
    a = int(0.5*len(biceps_env))
    biceps_MVC = np.max(biceps_env[0:a])
    triceps_MVC = np.max(triceps_env)
    deltoid_MVC  = np.max(deltoid_env)
    pecs_MVC = np.max(pecs_env)

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
    
    
    





































    
    
    
    
    
    