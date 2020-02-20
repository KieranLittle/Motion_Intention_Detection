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

def filter_smooth(old_dataset):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #N = 15
    #imu_smooth = np.convolve(old_dataset['imu'], np.ones((N,))/N, mode='valid')
    
    new_dataset = old_dataset.copy()
    new_dataset['imu'] = lowpass_filter(new_dataset['time'],new_dataset['imu'],2,2,100)
    new_dataset['stretch'] = lowpass_filter(new_dataset['time'],new_dataset['stretch'],1,8,100)
    
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
    
    count = 0
    #amplitude_var = np.zeros(len(dataset))
    amplitude_temp = []
    
    traj_segment ={}
    trajectory = {}
    
    time_3degrees = []
    time_half = []
    time_end = []
    
    peak_segment_velocity = []
    mean_segment_velocity = []
    segment_max_point = []
    
    segment_cut_off = []
    
    for j in range(0,len(dataset['time'])):
      if (dataset['time'][j] % 8.0 == 0.0) & (dataset['time'][j] != 0.0) & (dataset['time'][j] != 1200.0): #& (dataset['imu'][j] < 5):
          
          #amplitude_var[j] = amplitudes[count]
          trajectory[count] = dataset.iloc[j:j+400,:] #400
          
          amplitude_temp.append(np.max(trajectory[count]['imu']))      #amplitudes[count])
          max_amp = np.max(trajectory[count]['imu'])
          
          amp_range = trajectory[count].loc[(trajectory[count]['imu'] > 0.98*max_amp)]
          amp_range_index = amp_range.index[0]
          
          #max_amp_index = trajectory[count].index[trajectory[count]['imu'] == max_amp].tolist()
          
          trajectory[count] = dataset.iloc[j:amp_range_index+50,:]
          peak_velocity = np.max(np.diff(dataset[j:j+400]['imu'],1))/0.01
          
          #end_window = j+50
          
          a = trajectory[count].loc[(trajectory[count]['imu'] < (10.0+dataset['imu'][j]))]
          #a = dataset[j:j+1]
          # a = a.reset_index()
          
          duration_window = int(np.round(a.iloc[-1,0]-dataset['time'][j]))
          
          # for i in range(1,len(a)-1):
              
          #     if (trajectory[count].iloc[i,0] - trajectory[count].iloc[i-1,0] < 0)  & (trajectory[count].iloc[i+1,0] - trajectory[count].iloc[i,0] <0):
          #         a = a.drop(a.index[i], axis=0, inplace=True)
          
          
          #a_emg = trajectory[count].loc[(trajectory[count]['biceps'] > 0.0065)]
          
          #b = trajectory[count].loc[(trajectory[count]['time'] > (trajectory[count].iloc[0,0]+end_window/100))]
          #c = trajectory[count].loc[(trajectory[count]['imu'] > 0.9*amplitude_temp[count])]
          
          segment_cut_off.append(a.iloc[-1,0])
          segment_max_point.append(amp_range.iloc[0,0])
          
          time_3degrees.append(a.iloc[-1,0]- (trajectory[count].iloc[0,0]))
          #time_3degrees.append(a.iloc[0,0]-(trajectory[count].iloc[0,0]))
          #time_half.append(b.iloc[0,0]-(trajectory[count].iloc[0,0]))
          #time_end.append(c.iloc[0,0]-(trajectory[count].iloc[0,0]))
          time_end.append(dataset.iloc[amp_range_index+50,0]-trajectory[count].iloc[0,0])       #dataset[(amp_range_index+70):(amp_range_index+71)]['time'])            #amp_range.iloc[0,0]-(trajectory[count].iloc[0,0]))
          
          time_3degrees = [ round(elem, 2) for elem in time_3degrees ]
          
          #traj_segment[count] = dataset.iloc[a.index[-1]-duration_window:a.index[-1],:] #dataset.iloc[j:end_window,:]#dataset.iloc[a.index[0]-80:a.index[0],:]
          traj_segment[count] = dataset.iloc[j:a.index[-1],:]
          #mean_velocity = np.mean(np.diff(dataset[a.index[-1]-duration_window:amp_range_index+50]['imu'],1))/0.01
          mean_velocity = np.mean(np.diff(dataset[j:amp_range_index+50]['imu'],1))/0.01
           
          peak_segment_velocity.append(peak_velocity)
          mean_segment_velocity.append(mean_velocity)
          #####
          
          count=count+1
          
    peak_segment_velocity = [ round(elem, 2) for elem in peak_segment_velocity ]
    time_3degrees = [ round(elem, 2) for elem in time_3degrees ]
    time_end = [ round(elem, 2) for elem in time_end ]
#    time_half = [ round(elem, 2) for elem in time_half ]
#    time_end  = [ round(elem, 2) for elem in time_end  ]    
#          
    return traj_segment, trajectory, amplitude_temp, peak_segment_velocity, mean_segment_velocity, time_3degrees,time_half,time_end, segment_cut_off, segment_max_point
    
def plot_segments(segments, peak_amplitude,peak_velocity):
    
    import matplotlib.pyplot as plt
    
    for i in range(0,20):
        
        plt.figure(i,figsize=(15,4)) 
        
        plt.suptitle('Peak Velocity: ' +str(peak_velocity[i]) +' , Peak Amplitude: '+ str(peak_amplitude[i]))
        plt.subplot(1,6,1)
        plt.plot(segments[i]['time'],segments[i]['imu'])
        plt.ylim(0,10)
        
        plt.subplot(1,6,2)
        plt.plot(segments[i]['time'],segments[i]['stretch'])
#        plt.ylim(3.324,3.32)
        
        plt.subplot(1,6,3)
        plt.plot(segments[i]['time'],segments[i]['biceps'])
        #plt.ylim(0.005,0.015)
        
        plt.subplot(1,6,4)
        plt.plot(segments[i]['time'],segments[i]['triceps'])
        #plt.ylim(0.003,0.008)
        
        plt.subplot(1,6,5)
        plt.plot(segments[i]['time'],segments[i]['deltoid'])
        #plt.ylim(0.002,0.004)
        
        plt.subplot(1,6,6)
        plt.plot(segments[i]['time'],segments[i]['pecs'])
        #plt.ylim(0.003,0.005)
        
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
        
        time = trajectory[segment_num]['time']-trajectory[segment_num].iloc[0,0]
        
        plt.subplot(1,6,1)
        plt.plot(time,trajectory[segment_num]['imu'])
        plt.title('segment cuff off: {} and trajectory max point = {}'.format(segment_line[segment_num], segment_max_point[segment_num]))
        
        plt.subplot(1,6,2)
        plt.plot(time,trajectory[segment_num]['stretch'])
#        plt.ylim(3.324,3.32)
        
        plt.subplot(1,6,3)
        plt.plot(time,trajectory[segment_num]['biceps'])
        #plt.ylim(0.005,0.015)
        
        plt.subplot(1,6,4)
        plt.plot(time,trajectory[segment_num]['triceps'])
        #plt.ylim(0.003,0.008)
        
        plt.subplot(1,6,5)
        plt.plot(time,trajectory[segment_num]['deltoid'])
        #plt.ylim(0.002,0.004)
        
        plt.subplot(1,6,6)
        plt.plot(time,trajectory[segment_num]['pecs'])
        #plt.ylim(0.003,0.005)
        
        
        plt.show()
        
    return 0
    

def extract_features(segments, peak_amplitude, peak_velocities,  mean_segment_velocity,time_3degrees,time_half,time_end):
    
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  from scipy import stats
    
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

  for segment_number in range(0,len(segments)):

    stretch_sum.append(segments[segment_number]['stretch'].sum())
    imu_sum.append(segments[segment_number]['imu'].sum())
    biceps_sum.append(segments[segment_number]['biceps'].sum())
    triceps_sum.append(segments[segment_number]['triceps'].sum())
    deltoid_sum.append(segments[segment_number]['deltoid'].sum())
    pecs_sum.append(segments[segment_number]['pecs'].sum())
    #amplitude.append(amplitudes[segment_number])

    biceps_max.append(np.max(np.abs(segments[segment_number]['biceps'])))
    triceps_max.append(np.max(np.abs(segments[segment_number]['triceps'])))
    deltoid_max.append(np.max(np.abs(segments[segment_number]['deltoid'])))
    stretch_max.append(np.max(segments[segment_number].iloc[:-10,1]))
    imu_max.append(np.max(segments[segment_number]['imu']))

    #velocity_max.append(np.max(np.gradient(segments[segment_number]['imu'])))
    velocity_max.append(np.max(np.diff(segments[segment_number]['imu'],1))/0.01)
    #curr_velocity.append(np.diff(segments[segment_number]['imu'],1)[-1]/0.01)
    #curr_acc.append(np.diff(segments[segment_number]['imu'],2)[-1]/0.01)
    
    curr_velocity.append(np.mean(np.diff(segments[segment_number]['imu'],1)[(len(segments[segment_number])-30):]))
    curr_acc.append(np.mean(np.diff(segments[segment_number]['imu'],2)[(len(segments[segment_number])-30):]))
    
    #curr_biceps_change.append(np.diff(segments[segment_number]['biceps'],1)[-1]/0.01)
    curr_biceps_change.append(np.mean(np.diff(segments[segment_number]['biceps'],1)[(len(segments[segment_number])-30):]))
    curr_biceps_change2.append(np.mean(np.diff(segments[segment_number]['biceps'],2)[(len(segments[segment_number])-30):]))
    #curr_biceps_change2.append(np.diff(segments[segment_number]['biceps'],2)[-1]/0.01)
    
    curr_triceps_change.append(np.mean(np.diff(segments[segment_number]['triceps'],1)[(len(segments[segment_number])-30):]))
    curr_triceps_change2.append(np.mean(np.diff(segments[segment_number]['triceps'],2)[(len(segments[segment_number])-30):]))
    
    curr_stretch_change.append(np.mean(np.diff(segments[segment_number]['stretch'],1)[(len(segments[segment_number])-30):]))
    curr_stretch_change2.append(np.mean(np.diff(segments[segment_number]['stretch'],2)[(len(segments[segment_number])-30):]))

    stretch_diff.append(segments[segment_number].iloc[-1]['stretch'] - segments[segment_number].iloc[0]['stretch'])
    stretch_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,1],1)/0.01))
    biceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,3],1)/0.01))
    triceps_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,4],1)/0.01))
    deltoid_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,5],1)/0.01))
    pecs_gradient.append(max(np.diff(segments[segment_number].iloc[:-10,6],1)/0.01))
    
    #stretch_diff.append(segments[segment_number].iloc[-1]['stretch'] - segments[segment_number].iloc[0]['stretch'])

    imu_diff.append(segments[segment_number].iloc[-1]['imu'] - segments[segment_number].iloc[0]['imu'])
    biceps_diff.append(segments[segment_number].iloc[-1]['biceps'] - segments[segment_number].iloc[0]['biceps'])
    triceps_diff.append(segments[segment_number].iloc[-1]['triceps'] - segments[segment_number].iloc[0]['triceps'])
    deltoid_diff.append(segments[segment_number].iloc[-1]['deltoid'] - segments[segment_number].iloc[0]['deltoid'])
    
    biceps_min.append(np.min(segments[segment_number]['biceps']))
    biceps_min_time.append(segments[segment_number].loc[segments[segment_number].idxmin()['biceps']]['time']-segments[segment_number].iloc[0,0])
    
    triceps_min.append(np.min(segments[segment_number]['triceps']))
    deltoid_min.append(np.min(segments[segment_number]['deltoid']))
  #biceps_sum = [i**2 for i in biceps_sum]

  dict = {'curr_biceps_change2': curr_biceps_change2, 'curr_triceps_change2': curr_triceps_change2, 'triceps_min':triceps_min,'biceps_min':biceps_min,'biceps_sum':biceps_sum, 'biceps_max':biceps_max, 'imu_max':imu_max, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc, 'stretch_gradient':stretch_gradient} #'window_max_velocity':velocity_max, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc, 'biceps_diff':biceps_diff, 'imu_diff':imu_diff, 'biceps_min':biceps_min, 'biceps_min_time':biceps_min_time } #{'biceps_sum':biceps_sum, 'biceps_max':biceps_max, 'biceps_gradient':biceps_gradient, 'triceps_gradient':triceps_gradient,'stretch_gradient':stretch_gradient, 'imu_diff':imu_diff, 'curr_velocity':curr_velocity, 'curr_acc':curr_acc }
  #'deltoid_gradient':deltoid_gradient, 'pecs_gradient':pecs_gradient, 'imu_max':imu_max, 'stretch_gradient':stretch_gradient,'stretch_max':stretch_max} # 'stretch_gradient':stretch_gradient, 'window_max_velocity':velocity_max'stretch_gradient':stretch_gradient 'imu_diff':imu_diff 'window_max_velocity':velocity_max}  #'stretch_diff':stretch_gradient #stretch_diff':stretch_diff} #'velocity_max':velocity_max, 'biceps_max':biceps_max, 'triceps_diff':triceps_diff, 'deltoid_diff':deltoid_diff 'velocity_max':velocity_max 'stretch_diff':stretch_diff,'imu_diff':imu_diff,  'triceps_sum':triceps_sum,'deltoid_sum':deltoid_sum,'pecs_sum':pecs_sum,'amplitude':amplitude}
  
  df2 = pd.DataFrame(dict) 
  
  df2['End Window Time'] = time_3degrees
   
  #df['peak_segment_velocity'] = peak_segment_velocity
  df2['Peak Amplitude'] = peak_amplitude
  df2['Peak Velocity'] = peak_velocities 
  df2['Mean Velocity'] = mean_segment_velocity

  #df['Time at half traj'] = time_half
  df2['Time at end'] = time_end
  
  df2 = df2[(np.abs(stats.zscore(df2)) < 4).all(axis=1)]

  sns.pairplot(df2)
  plt.show()
  
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
    
    X_org = df.drop(['Peak Amplitude','Peak Velocity','Mean Velocity','Time at end'],axis=1)
    
    #y = df['amplitude']
    y = df[['Peak Velocity','Time at end','Peak Amplitude','Mean Velocity']]
    
    # Split the data into train and test randomly with the test size = 30%, stratify data to split classification evenly
    X_train, X_test, y_train, y_test = train_test_split(X_org, y, test_size=0.30, random_state=42) #random_state=42
    
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
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    
    clf = SVR(kernel = 'rbf',C= 10, gamma = 1, epsilon=1,degree=2) # 10,100
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
    
    #print(clf.score(velocity_predictions,y_test_velocity))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_velocity, velocity_predictions)/np.mean(velocity_predictions))
    #print(clf.coef_)
    
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_velocity_predictions)
    plt.plot([0,200],[0,200],'r',lw=1)
    
    #plt.scatter(y_test,predictions)
    plt.show()
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler2 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler2.fit_transform(X_train)
    X_test_scaled = min_max_scaler2.transform(X_test)
    
    clf2 = SVR(kernel = 'rbf',C=10,gamma=0.1,epsilon=0.01,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    clf2.fit(X_train_scaled,y_train_time)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    time_predictions = clf2.predict(X_test_scaled)
    df_time_predictions = pd.DataFrame({'y_test':y_test_time, 'predictions':time_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_time_predictions)
    plt.plot([1.5,4.5],[1.5,4.5],'r',lw=1)
    
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('Duration')
    print('MAE:', metrics.mean_absolute_error(y_test_time, time_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_time, time_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_time, time_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_time, time_predictions)/np.mean(time_predictions))
    
    #%%%%%%%%%%% AMPLTIUDE %%%%%%%%%%%%%%
        # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler2 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler2.fit_transform(X_train)
    X_test_scaled = min_max_scaler2.transform(X_test)
    
    clf2 = SVR(kernel = 'rbf',C=10,gamma=0.1,epsilon=0.01,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    clf2.fit(X_train_scaled,y_train_amplitude)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    amplitude_predictions = clf2.predict(X_test_scaled)
    df_amplitude_predictions = pd.DataFrame({'y_test':y_test_amplitude, 'predictions':amplitude_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_amplitude_predictions)
    plt.plot([20,100],[20,100],'r',lw=1)
    
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('Amplitude')
    print('MAE:', metrics.mean_absolute_error(y_test_amplitude, amplitude_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_amplitude, amplitude_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_amplitude, amplitude_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_amplitude, time_predictions)/np.mean(amplitude_predictions))
    
    
     #%%%%%%%%%%% MEAN VEL %%%%%%%%%%%%%%
    
    # Scale the train data to range [0 1] and scale the test data according to the train data
    min_max_scaler2 = preprocessing.MinMaxScaler()
    X_train_scaled = min_max_scaler2.fit_transform(X_train)
    X_test_scaled = min_max_scaler2.transform(X_test)
    
    clf2 = SVR(kernel = 'rbf',C=10,gamma=0.1,epsilon=0.01,degree=1) # 'rbf',C= 1000, gamma = 0.01, epsilon=0.01,degree=2) #100,1000,10,1
    clf2.fit(X_train_scaled,y_train_mean_vel)
    
    # The coefficients
    #print('Coefficients: \n', lm.coef_)
    
    mean_vel_predictions = clf2.predict(X_test_scaled)
    df_mean_vel_predictions = pd.DataFrame({'y_test':y_test_mean_vel, 'predictions':mean_vel_predictions})
    
    #plt.scatter(y_test,predictions)
    sns.lmplot(x= 'y_test',y = 'predictions', data = df_mean_vel_predictions)
    plt.plot([5,50],[5,50],'r',lw=1)
    
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    
    print('Mean Velocity')
    print('MAE:', metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions))
    print('MSE:', metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_mean_vel, mean_vel_predictions)))
    print('% Error: ', 100*metrics.mean_absolute_error(y_test_mean_vel, mean_vel_predictions)/np.mean(mean_vel_predictions))
    
    X_test2 = X_test.copy() 
    X_test2['peak_velocity_predictions'] = velocity_predictions
    X_test2['duration_predictions'] = time_predictions
    X_test2['amplitude_predictions'] = amplitude_predictions
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
        setpoint = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]*df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-2]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]
        frequency = 100
        time = df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]#(df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-1]-current)/df_predictions_SVM.iloc[segment_num,len(df_predictions_SVM.T)-2]
        
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
    
    plt.subplot(2,2,1)
    plt.plot(emg_MVC['Time'], emg_MVC['Biceps'])
    plt.title('biceps')
    
    plt.subplot(2,2,2)
    plt.plot(emg_MVC['Time'], emg_MVC['Triceps'])
    plt.title('triceps')
    
    plt.subplot(2,2,3)
    plt.plot(emg_MVC['Time'], emg_MVC['Deltoid'])
    plt.title('Deltoid')
    
    plt.subplot(2,2,4)
    plt.plot(emg_MVC['Time'], emg_MVC['Pecs'])
    plt.title('Pecs')
    
    plt.show()
    
    biceps_env = filteremg(emg_MVC['Time'], emg_MVC['Biceps'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=1)
    triceps_env = filteremg(emg_MVC['Time'], emg_MVC['Triceps'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=1)
    deltoid_env = filteremg(emg_MVC['Time'], emg_MVC['Deltoid'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=1)
    pecs_env = filteremg(emg_MVC['Time'], emg_MVC['Pecs'], low_pass=10, sfreq=1000, high_band=20, low_band=450, graphs=1)
    
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



    
    
    
    
    
    