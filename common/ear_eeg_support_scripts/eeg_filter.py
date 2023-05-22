import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from scipy import signal



# Filter the stuff
def filter_eeg(reref_data, hpfilt_enable, lpfilt_enable, fs):
    
    filt_data = reref_data
    
    # highpass filter
    if hpfilt_enable == True:
        sos = signal.butter(5, 0.05, 'hp', fs=1000, output='sos')
        filt_data = signal.sosfilt(sos, filt_data)
        
    # lowpass filter
    if lpfilt_enable == True:
        sos = signal.butter(5, 50, 'lp', fs=1000, output='sos')
        filt_data = signal.sosfilt(sos, filt_data)
    
    return filt_data


def filter_studies(all_raw_data):
    num_cores = multiprocessing.cpu_count()
    filtered_data = []

    for i in range(len(all_raw_data)):
        study = all_raw_data[i]
        num_chs = study.shape[1]
        filt_data = np.array(Parallel(n_jobs=num_cores)(delayed(filter_eeg)(study[:,ch], True, True, 1000) for ch in range(0,num_chs)))    

        #filtered_study = filter_eeg(study, True, True, 1000)
        filtered_data.append(filt_data.T)
    
    
    #filtered_data = np.array(filtered_data)


    return filtered_data