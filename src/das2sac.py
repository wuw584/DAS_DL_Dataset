import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import os
from scipy import signal
from obspy import Stream, Trace
import gc

from nptdms import TdmsFile
from DasTools import DasPrep as dp

import pyproj

from multiprocessing import Pool
from functools import partial

from das2sac_batch_run_parallel_func import das_processing, get_das_file_time, das_st_write_sac
def get_continuous_segments(fname_datetime, file_len, tol):
    fname_datetime_diff = np.diff(fname_datetime)
    file_len_timedelta = datetime.timedelta(seconds=file_len*1.0001)
    segment_diff = np.where(fname_datetime_diff > file_len_timedelta)[0]  # define continuous segments by no files seperated more than the file length (15s)
    segment_start = np.r_[0, segment_diff + 1]
    segment_end = np.r_[segment_diff, len(fname_datetime) - 1]

    segment_start_datetime = fname_datetime[segment_start] + file_len_timedelta * 1.5 # shift to later by 1.5 file length to give buffer for rolling 
    segment_end_datetime = fname_datetime[segment_end] - file_len_timedelta * 1.5  # shift to earlier by 1.5 file length to give buffer for rolling 

    continuous_segment_size = np.array([x.total_seconds() for x in (segment_end_datetime - segment_start_datetime)])
    
    segment_choose = np.where(continuous_segment_size > tol)[0] 
    return segment_start_datetime[segment_choose], segment_end_datetime[segment_choose], continuous_segment_size[segment_choose]




dir0 = '/home/disk/disk01/wzm/DAS_DL_Dataset/data/taiwan/Micro_Seis'
day_list = np.sort( [f for f in os.listdir(dir0) if "M" in f])
print(len(day_list))
for day in day_list[1:]:

    datapath =  '/home/disk/disk01/wzm/DAS_DL_Dataset/data/taiwan/Micro_Seis/'+day +'/'
    sacpath = '/home/disk/disk01/wzm/DAS_DL_Dataset/data/taiwan/Micro_SAC_400Hz/'+day+'/'
    datafile = glob.glob(datapath+'*.h5')
    datafile.sort()
    fname_format = datapath + 'XFJ_23km_GL_10m_frq_400Hz_sp_4m_UTC_%Y%m%d_%H%M%S.%f.h5'
    fname_npdatetime = np.array([np.datetime64(datetime.datetime.strptime(x, fname_format),'us') for x in datafile])
    fname_datetime = np.array([datetime.datetime.strptime(x, fname_format) for x in datafile])

    if not os.path.exists(sacpath):  os.makedirs(sacpath)

        
    metadata = dp.read_das(datafile[len(datafile)//2], metadata=True)

    for key in metadata.keys():
        print(key, ':', metadata[key])
        

    dt = metadata['dt']
    nt = metadata['nt']
    file_len = dt*nt

    ch1 = 0
    ch2 = 5632
    das_ch_id = np.arange(ch1, ch2)

    print(dt,nt, file_len)

    segment_start_datetime, segment_end_datetime, continuous_segment_size = get_continuous_segments(fname_datetime, file_len, tol=10*60) # segments lasting more than 20 min
    print(continuous_segment_size)

    for isegment in range(len(segment_start_datetime)):
        start_time, end_time = segment_start_datetime[isegment], segment_end_datetime[isegment]
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        segment_size = end_time - start_time
        
        tmp = segment_size.astype('timedelta64[h]')
        print(f'Segment {isegment} : {tmp}')
        print(start_time)
        print(end_time)
        print(' ' )



    nw = 'XFJ'
    sta = 'DAS'
    mlist = np.array([1])
    nprocs = 2

    chunk_size = np.timedelta64(7200, 's') #increment in hours
    increment = np.timedelta64(30, 's') # one-time increment of data one thread holds


    interval = increment * nprocs # total increment of data all threads hold
    for isegment in range(len(segment_start_datetime)):
        
        start_time, end_time = segment_start_datetime[isegment], segment_end_datetime[isegment]
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        segment_size = end_time - start_time
        chunk_num = segment_size // chunk_size + 1
        
        print('Segment id: %d'%(isegment))
        print('Segment size: %s'%(segment_size.astype('timedelta64[s]'))) 
        print('Regular chunk size: %s'%(chunk_size))
        print('Regular chunk number: %s'%(chunk_num - 1)) 
        print('Remainder chunk size: %s'%((segment_size % chunk_size).astype('timedelta64[s]')))
        
        print('nCPU number: %s'%(nprocs)) 
        print('Increment by each worker: %s'%(increment)) 
        print('Interval size: %s'%(interval)) 
        print('Interval number of regular chunks (if any): %s'%(chunk_size // interval))
        print('Interval number of the remainder chunk: %s'%(segment_size % chunk_size // interval))
        print(' ')
        
        segment_folder_path = sacpath + 'SAC-segment-' + ''.join(str(start_time.astype('datetime64[s]')).split(':')) 
        if not os.path.exists(segment_folder_path):
            os.makedirs(segment_folder_path)

        for ichunk in range(chunk_num):

            chunk_start_time = start_time + chunk_size * ichunk

            chunk_folder_path = os.path.join(segment_folder_path, 
                                            'SAC-chunk-' + ''.join(str(chunk_start_time.astype('datetime64[s]')).split(':')))

            if os.path.exists(chunk_folder_path):
                print('Chunk %d of Segment %d folder already exists: %s'%(ichunk, isegment, chunk_folder_path))
                print(' ')
                continue
                
            print('Chunk %d in Segment %d: %s'%(ichunk, isegment, chunk_folder_path))
            print('Start time of this chunk: %s'%(chunk_start_time.astype('datetime64[s]')))

            data = []
            interval_num = chunk_size // interval if ichunk < chunk_num-1 else segment_size % chunk_size // interval
            print('Total intervals in this Chunk: %d'%(interval_num))
            with Pool(processes=nprocs) as pool:
                for j in range(int(interval_num)):
                    print('Working on Interval %d in Chunk %d in Segment %d'%(j, ichunk, isegment))
                    increment_start_times = [ chunk_start_time + (interval * j) + (increment * i) for i in range(nprocs) ]
                    res = pool.map(partial(das_processing, 
                                        interval=increment, 
                                        datafile=datafile, 
                                        datafile_time=fname_npdatetime, 
                                        ch1=ch1, ch2=ch2, mlist=mlist), 
                                increment_start_times)

                    data.append(np.concatenate([res[i][0] for i in range(len(res))], axis=1))
                    dt = res[0][1]
                    del res

            if len(data) > 0:
                data = np.concatenate(data, axis=1)

                chunk_start_time_1 = chunk_start_time + increment
                datafile_arg_choose = np.where((fname_npdatetime>=chunk_start_time)&(fname_npdatetime<chunk_start_time_1))[0]
                chunk_start_time_from_file = fname_npdatetime[datafile_arg_choose][0]

                das_st = Stream()
                for ich in das_ch_id:

                    data_ich = np.where(das_ch_id==ich)[0][0]

                    tr = Trace(data=data[data_ich,:], header={'network':nw, 
                                                            'station': sta, 
                                                            'location':str(data_ich), 
                                                            'channel': str(ich),
                                                            'starttime':str(chunk_start_time_from_file), 
                                                            'delta':dt})

                    das_st.append(tr)

                print('Writing to sac...')
                
                if not os.path.exists(chunk_folder_path):
                    os.makedirs(chunk_folder_path)
                
                with Pool(processes=nprocs) as pool:
                    pool.map(partial(das_st_write_sac, date_folder_path=sacpath, write_coordinates=False), das_st)

                del tr
                del das_st
                del data
            gc.collect()
            print(' ')