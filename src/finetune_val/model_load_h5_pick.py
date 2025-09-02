import os
from pathlib import Path
import obspy
from diting import DiTing_EQDet_PhasePick_predict
from obspy import Stream , Trace
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import DasPrep as dp
import datetime
import obspy
from scipy import signal
import pandas as pd


output_dir = "/home/disk/disk02/wzm/DAS_DL_Dataset/course-xiao/course_sft_code_enc/course_script/debug_log/val_3km_npy/"

eq_cat = pd.read_csv("/home/disk/disk02/wzm/DiTingProject/results/xfj23km/xfj3km/DetectedFinal.dat", sep='\s+')


eq_time = np.array([datetime.datetime.strptime(str(eq_cat['Date'].values[i])+' '+str(eq_cat['Time'].values[i]), '%Y/%m/%d %H:%M:%S.%f') 
       for i in range(len(eq_cat))])
# for i_day in range(0,len(day_list) , 10):
#     data = np.concatenate([dp.read_das(day_list[i_day+ i]) for i in range(12) ] ,axis= 1)
#     start_time = datetime.datetime.strptime(' '.join(day_list[i_day].split('_')[-2:]) , '%Y%m%d %H%M%S.%f.h5')
for npy in range(147,408):
    
    data = np.load("/home/disk/disk02/wzm/DAS_DL_Dataset/data/xfj/das_event_reorganize/xfj_das_re_eq_"+str(npy)+".npy")
    start_time = eq_time[npy] - datetime.timedelta(seconds= 20)
    nch = data.shape[0] 
    nt = data.shape[1]
    dt = 1./300
    data = signal.decimate(data , 3, axis = -1).astype('float32')
    dt = 1./100
    start = time.time()
    print(nt,nch)
    for ch in range(0,nch,3)[:-1]:     
        stream = Stream()
        stream.append( Trace(data=data[ch,:] , header = {'network':'xfj', 
                                                'station': 'DAS', 
                                                'location':str(ch), 
                                                'channel': 'HHE',
                                                'starttime':str(start_time), 
                                                'delta':dt}))
        stream.append( Trace(data=data[ch+1,:] , header = {'network':'xfj', 
                                                'station': 'DAS', 
                                                'location':str(ch+1), 
                                                'channel': 'HHN',
                                                'starttime':str(start_time), 
                                                'delta':dt}))
        stream.append( Trace(data=data[ch+2,:] , header = {'network':'xfj', 
                                                'station': 'DAS', 
                                                'location':str(ch+2), 
                                                'channel': 'HHZ',
                                                'starttime':str(start_time), 
                                                'delta':dt}))
        stream.merge()
        stream.detrend('demean')
        # print( stream[0].stats)
        # 不是必须的，尤其做远震可以用比较低的采样率，比如20Hz之类的，这里只是提醒下采样率的影响
        # if stream[0].stats.sampling_rate != 100:

        # 强制对齐起始终止
        stream.trim(starttime=stream[0].stats.starttime, endtime=stream[0].stats.endtime,pad=True, fill_value=0)
        starttime = stream[0].stats.starttime
        events, confidence, num = DiTing_EQDet_PhasePick_predict(stream=stream, p_th=0.01, s_th=0.01, det_th=0.05)
        # events, _ = DiTing_EQDet_PhasePick_predict_for_mulit_events(stream, device, model, window_length=10000, step_size=3000, p_th=0.1, s_th=0.1, det_th=0.3)
        
        ptime = []
        stime = []
        s_ptime = []
        model_name = 'DiTing_fintune_3k'
        data_set_name = ''

        save_waveforms = False
        before_P = 5
        after_S = 15

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        example_txt_file = open(output_dir+ str(npy)+'.txt', 'a+')

        new_events = []

        for t_ev in events:
            P = t_ev[1][0]
            S = t_ev[2][0]
            # 筛选条件和保存根据需求改一下
            if np.isnan(P[0]) or np.isnan(S[0]):
                continue
            # print(P, S)
            # convert to UTCDateTime
            P[0] = (obspy.UTCDateTime(starttime) + P[0]/stream[0].stats.sampling_rate)
            S[0] = (obspy.UTCDateTime(starttime) + S[0]/stream[0].stats.sampling_rate)
            s_ptime.append((S[0]) - (P[0]))
            ptime.append(P[0].datetime)
            stime.append(S[0].datetime)
            new_events.append(t_ev)

            line = '{} {} {} {} {} {}\n'.format(stream[0].stats.station,ch , P[0], P[1], S[0], S[1])
            example_txt_file.write(line)
        example_txt_file.close()

        events = new_events

    end_time_2 = time.time()
    execution_time = end_time_2 - start
    print(f"执行时间：{execution_time} 秒")

# conda activate diting
# cd /home/disk/disk02/wzm/DAS_DL_Dataset/course-xiao/course_sft_code_enc
# python diting/diting_server.py
# cd /home/disk/disk02/wzm/DAS_DL_Dataset/src/finetune_val