import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import os
from scipy import signal
import EQNet.docs.DasPrep_n as dp

from nptdms import TdmsFile
import EQNet.docs.DasPrep_n as idas

workpath = '/home/wuzm/data/DAS_DL_Dataset/data/xfj/das_event_reorganize/'

eq_catalog_file = '/home/wuzm/data/DAS_DL_Dataset/data/xfj/DetectedFinal.dat'
eq_cat = pd.read_csv(eq_catalog_file, delim_whitespace=True)


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

eq_cat[mag >-2]
 # strong mag
import matplotlib.animation as animation
import das_event_manual_pick as das_mp
from scipy import signal
import EQNet.docs.DasPrep_n as dp


dt_before, dt_after = 20, 60

save_file_name_prefix = workpath + 'xfj_das_re_eq_'

das_file = glob.glob(save_file_name_prefix+'*.npy')
for iev in range(168  , 169):
    savename = save_file_name_prefix + str(iev) + '.npy'
    if savename in das_file:
              print(iev)
              fq = 300
              dt = 1./300.
              print(savename)
              data_raw = np.load(savename)
              print(data_raw.shape)

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
              print(data.shape)

              # # plot hot map to find the arrive time
              # # https://stackoverflow.com/questions/39079562/matplotlib-animation-vertical-cursor-line-through-subplots
              # # fig, ax = plt.subplots(figsize=[8,7])

              clim = data.std() 

              # diting_event = pd.read_csv("/home/wuzm/data/DAS_DL_Dataset/data/xfj/sac_diting/xfj_das_re_eq_" + str( iev ) + ".xfj.DAS._P_S.txt" , sep='\s+',header=None, names=['sta','chn','P0','P1','S0','S1'])
              diting_event = pd.read_csv("/home/disk/disk01/wzm/DAS_DL_Dataset/data/xfj/npy_diting_100Hz/" + str( iev ) + ".txt" , sep='\s+',header=None, names=['sta','chn','P0','P1','S0','S1'])

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
              print(p_time)
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

              savefigname = "/home/disk/disk01/wzm/DAS_DL_Dataset/data/xfj/sac_diting_fig_multi_100Hz/" + 'xfj_das_diting_eq_' + str(iev)+ '.png'
              fig.savefig(savefigname, dpi=300)

                     





              fig = plt.figure(figsize=[8,8])
              plt.imshow(data[:, :].T, aspect='auto', cmap ='seismic', 
                     vmin = -clim, vmax = clim,
                     extent=[0,data.shape[0], data.shape[1]*dt, 0])
              plt.xlabel('Channel')

              csv_file = glob.glob("/home/disk/disk01/wzm/DAS_DL_Dataset/data/xfj/picks_phasenet_das_raw_2/*csv")
              csv_file.sort()
              print(len(csv_file))
              def get_das_file_time(das_filename):
                     das_file_time_str = ' '.join(os.path.splitext(os.path.basename(das_filename))[0].split('_')[-2:])
                     return datetime.datetime.strptime(das_file_time_str, '%Y%m%d %H%M%S.%f')

              csv_file_time = np.array([get_das_file_time(csv_file[i]) for i in range(len(csv_file))])
              ev_time_before = eq_time[iev] - datetime.timedelta(seconds=dt_before)
              ev_time_after  = eq_time[iev] + datetime.timedelta(seconds=dt_after)
              # print(csv_file_time , ev_time_before ,ev_time_after )

              ins_start = np.searchsorted(csv_file_time,ev_time_before) - 1
              ins_end = np.searchsorted(csv_file_time, ev_time_after)

              das_file_time_select = csv_file_time[ins_start:ins_end]
              das_file_select = csv_file[ins_start:ins_end]

              for icsv in range(len(das_file_select)):
                     csv = das_file_select[icsv]
                     print(csv)

                     if os.path.getsize(csv) == 0:
                            print("文件为空")
                            continue
                     else:
                            picks = pd.read_csv(csv)
                            delta_time =(ev_time_before.__sub__(das_file_time_select[icsv])).total_seconds()
                            print(delta_time)
                            color = picks["phase_type"].map({"P": "r", "S": "b"})
                            plt.scatter(picks["channel_index"].values*10  , picks["phase_index"].values*dt - delta_time , c=color, s=1)
                     plt.scatter([], [], c="r", label="P")
                     plt.scatter([], [], c="b", label="S")


                     plt.legend()
                     plt.ylim(80,0)

                     arglist = ['Date', 'Time', 'Lat.', 'Lon.', 'Mag.', 'Dep.']
                     plt.text(2050, 75, eq_cat.iloc[iev][arglist])
                     plt.title("1-20Hz Bandpass")
                     plt.show()

                     savefigname = "/home/disk/disk01/wzm/DAS_DL_Dataset/data/xfj/sac_phasenetdas_fig_2/" + 'xfj_das_diting_eq_' + str(iev)+ '.png'
                     fig.savefig(savefigname, dpi=300)