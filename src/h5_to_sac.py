from obspy import Stream, Trace ,signal
import numpy as np
import glob
import EQNet.docs.DasPrep_n as dp
import datetime
def das_st_write_sac(das_tr, date_folder_path, write_coordinates=False):
    chn = int(das_tr.stats.location)
    if chn%3 == 0:
        das_tr.stats.channel = 'HHE'
    elif chn%3 == 1:
        das_tr.stats.channel = 'HHN'
    elif chn%3 == 2:
        das_tr.stats.channel = 'HHZ'
    if write_coordinates:
        das_tr.stats.sac = {'stla': das_tr.stats.coordinates['latitude'], 
                      'stlo': das_tr.stats.coordinates['longitude']}
    nw = das_tr.stats.network
    sta = das_tr.stats.station
    
    das_tr.write(date_folder_path + '//'+'xfj_das_re_eq_' +'.'+nw+'.'+sta+'.'+str(chn//3*3)+'.'+das_tr.stats.channel+'.sac', format='SAC') 

save_file_dir 
 
das_files = glob.glob(data_path+'*.h5')

for das_f in das_files:
    fq = 300
    dt = 1./300.
    data_raw = dp.read_das(das_f)

    #  raw data and down sample
    data_raw_d = signal.decimate(data_raw, 3, axis=-1).astype('float32')
    dt *= 3
    fq = 100 


    ch1 = 0
    ch2 = 2944
    das_ch_id = np.arange(ch1, ch2)

    nw = 'xfj'
    sta = 'DAS'
    start_time = datetime.datetime.strptime(' '.join( das_f.split('_')[-2:]), '%Y%m%d %H%M%S.%f.h5')

    for ich in das_ch_id:
        data_ich = np.where(das_ch_id==ich)[0][0]

        tr = Trace(data=data_raw_d[ich,:], header={'network':nw, 
                                            'station': sta, 
                                            'location':str(data_ich), 
                                            'channel': str(ich),
                                            'starttime':str(start_time), 
                                            'delta':dt})
        
        das_st_write_sac(tr ,save_file_dir)

        