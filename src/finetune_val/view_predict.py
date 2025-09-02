import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import os
from scipy import signal
import DasPrep as dp


DAS_DL_Dataset_proj = "/home/disk/disk02/wzm/DAS_DL_Dataset"
workpath = DAS_DL_Dataset_proj+'/data/xfj/das_event_reorganize/'


diting_result_dir = DAS_DL_Dataset_proj+"/course-xiao/course_sft_code_enc/course_script/debug_log_4w/val_3km_npy/"
savefig_dir = DAS_DL_Dataset_proj+"/course-xiao/course_sft_code_enc/course_script/debug_log_4w/val_3km_npy/fig/" 


eq_catalog_file = DAS_DL_Dataset_proj + '/data/xfj/DetectedFinal.dat'
eq_cat = pd.read_csv(eq_catalog_file, sep='\s+' )


eq_time = np.array([datetime.datetime.strptime(str(eq_cat['Date'].values[i])+' '+str(eq_cat['Time'].values[i]), '%Y/%m/%d %H:%M:%S.%f') 
       for i in range(len(eq_cat))])

# eq_time = eq_time - datetime.timedelta(seconds=8*3600)   # this catalog was in UTC

lat = eq_cat['Lat.'].values
lon = eq_cat['Lon.'].values
mag = eq_cat['Mag.'].values
dep = eq_cat['Dep.'].values
Coef = eq_cat['Coef.'].values

ev_files = glob.glob(workpath+'*.npy')
get_number_in_file =  lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
ev_files.sort(key=get_number_in_file)
print(np.mean(mag))


dt_before, dt_after = 20, 60

save_file_name_prefix = workpath + 'xfj_das_re_eq_'

das_file = glob.glob(save_file_name_prefix+'*.npy')
for iev in range(273, 408):
       savename = save_file_name_prefix + str(iev) + '.npy'
       if savename in das_file:
              print(iev)
              fq = 300
              dt = 1./300.
              print(savename)
              data_raw = np.load(savename)

              # down sample and save raw data 
              data_raw_d = signal.decimate(data_raw, 3, axis=-1).astype('float32')
              dt *= 3
              fq = 100 
              # data process
              data_p = dp.das_preprocess(data_raw_d)
              # f1, f2 = 15, 35
              f1, f2 = 1,  20
              data = dp.bandpass(data_p * signal.windows.tukey(data_p.shape[1], alpha=0.2), dt, fl=f1, fh=f2)
              # data = dp.lowpass(data_p * signal.windows.tukey(data_p.shape[1], alpha=0.2), dt,  fh=10)
              # data = data_p

              # # plot hot map to find the arrive time
              # # https://stackoverflow.com/questions/39079562/matplotlib-animation-vertical-cursor-line-through-subplots
              # # fig, ax = plt.subplots(figsize=[8,7])

              clim = data.std() 

              diting_event = pd.read_csv(diting_result_dir   + str( iev ) + ".txt" , sep='\s+',header=None, names=['sta','chn','P0','P1','S0','S1'])

              start_time = eq_time[iev]
              fig = plt.figure(figsize=[8,8])
              plt.imshow(data[:, :].T, aspect='auto', cmap ='seismic', 
                     vmin = -clim, vmax = clim,
                     extent=[0,data.shape[0], data.shape[1]*dt, 0])
              plt.xlabel('Channel')
              plt.ylabel('Time (s)')
              p_time = np.array([datetime.datetime.strptime(diting_event["P0"][i], '%Y-%m-%dT%H:%M:%S.%fZ').__sub__(start_time)/datetime.timedelta(seconds=1)
                     for i in range(len(diting_event))])
              s_time = np.array([datetime.datetime.strptime(diting_event["S0"][i], '%Y-%m-%dT%H:%M:%S.%fZ').__sub__(start_time)/datetime.timedelta(seconds=1)
                     for i in range(len(diting_event))])
              p_s = np.array([i+ 20 for i in p_time])
              s_s = np.array([i +20  for i in s_time])

              # for i in range(len(diting_event)):
              #     print(datetime.datetime.strptime(diting_event["P0"][i], '%Y-%m-%dT%H:%M:%S.%fZ').__sub__(start_time).seconds)
              plt.scatter(diting_event["chn"].values,p_s, c="r", s=1)
              plt.scatter(diting_event["chn"].values,s_s, c="b", s=1)
              plt.scatter([], [], c="r", label="P")
              plt.scatter([], [], c="b", label="S")

              plt.legend()
              arglist = ['Date', 'Time', 'Lat.', 'Lon.', 'Mag.', 'Dep.']
              plt.text(2050, 75, eq_cat.iloc[iev][arglist])
              plt.title("1-20Hz Bandpass")

              savefigname = savefig_dir + 'xfj_3km_das_diting_eq_' + str(iev)+ '.png'
              print(savefigname)
              fig.savefig(savefigname, dpi=100)

                     


