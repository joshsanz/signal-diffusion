import math
import h5py
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

def read_in_ear_eeg(spreadsheet, data_filepath, input_users, data_chs, fs, plot_raw_data_enable):

    # read experiment details from spreadsheet
    details_arr, details_tags = read_spreadsheet(spreadsheet, input_users)
    
    # for each trial in spreadsheet
    for t in range(0, len(details_arr[0:])):
        print('reading ' + details_arr[t][details_tags.get('filename')])
        
        # read in ear eeg data
        raw_data = get_ear_eeg(details_arr[t][details_tags.get('filename')], details_arr[t][details_tags.get('file extention')], data_filepath, details_arr[t][details_tags.get('experiment length')], data_chs, fs)

        # plot raw eeg data
        if plot_raw_data_enable:
            for y in range(0,len(raw_data[0])):    
                plt.plot(range(0,len(raw_data[:,0])),raw_data[:,y])
                plt.title(details_arr[t][details_tags.get('filename')] + ', ch = ' + str(data_chs[y]))
                #plt.ylim((-0.0005, 0.0005))
                plt.show()
        
        # concatenate data from all experiments and details necessary for classification
        if t==0:
            all_raw_data = raw_data.T
            filenames = [details_arr[t][details_tags.get('filename')]]
            data_lengths = [len(raw_data)]
            file_users = [details_arr[t][details_tags.get('user')]]
            refs = [details_arr[t][details_tags.get('ref')]]
        else:
            all_raw_data = np.concatenate((all_raw_data ,raw_data.T), axis=1)
            filenames.append(details_arr[t][details_tags.get('filename')])
            data_lengths.append(len(raw_data))
            file_users.append(details_arr[t][details_tags.get('user')])
            refs.append(details_arr[t][details_tags.get('ref')])
            
    return all_raw_data, filenames, data_lengths, file_users, refs    

########################################################################################################################
# Read in experiment details from a spreadsheet (specifcying filenames, user, reference electrode, length of experiment and save them to clean
def read_clean_and_save(spreadsheet, data_filepath, input_users, data_chs, fs, plot_raw_data_enable):

    # read experiment details from spreadsheet
    details_arr, details_tags = read_spreadsheet(spreadsheet, input_users)
    
    # for each trial in spreadsheet
    for t in range(0, len(details_arr[0:])):
        filename = details_arr[t][details_tags.get('filename')]
        extension = details_arr[t][details_tags.get('file extention')]
        save_filepath = data_filepath[:-1] + '/../ear_eeg_clean/' + filename + '.h5'

        print('reading ' + filename)
        
        # read in ear eeg data
        raw_data = get_ear_eeg(filename, extension, data_filepath, details_arr[t][details_tags.get('experiment length')], data_chs, fs)
        
        
        print('saving clean in' + save_filepath)
        h5f = h5py.File(save_filepath, 'w')
        h5f.create_dataset('clean_data', data=raw_data)
        h5f.close() 
    return    

########################################################################################################################
# Read in experiment details from a spreadsheet and the clean data from high density files
def read_in_clean_data(spreadsheet, data_filepath, input_users, data_chs, fs, plot_raw_data_enable):
    # read experiment details from spreadsheet
    details_arr, details_tags = read_spreadsheet(spreadsheet, input_users)
    
    # for each trial in spreadsheet
    for t in range(0, len(details_arr[0:])):

        # Arya add start
        if t > 21:
            break
        # Arya end

        filename = details_arr[t][details_tags.get('filename')]
        print('loading clean ' + filename)
    
        # read in ear eeg data
        h5f = h5py.File(data_filepath+filename+'.h5','r')
        clean_data = h5f['clean_data'][:]
        h5f.close()

        # plot raw eeg data
        if plot_raw_data_enable:
            for y in range(0,len(clean_data[0])):    
                plt.plot(range(0,len(clean_data[:,0])),clean_data[:,y])
                plt.title(details_arr[t][details_tags.get('filename')] + ', ch = ' + str(data_chs[y]))
                #plt.ylim((-0.0005, 0.0005))
                plt.show()
        
        # concatenate data from all experiments and details necessary for classification
        if t==0:
            """
            all_raw_data = clean_data.T
            """
            # arya add start
            all_raw_data = []
            all_raw_data.append(clean_data)

            # all_raw_data = np.concatenate((all_raw_data ,clean_data.T), axis=1)
            #arya add end
            filenames = [details_arr[t][details_tags.get('filename')]]
            data_lengths = [len(clean_data)]
            file_users = [details_arr[t][details_tags.get('user')]]
            refs = [details_arr[t][details_tags.get('ref')]]
        else:
            # all_raw_data = np.concatenate((all_raw_data ,clean_data.T), axis=1)

            all_raw_data.append(clean_data)

            filenames.append(details_arr[t][details_tags.get('filename')])
            data_lengths.append(len(clean_data))
            file_users.append(details_arr[t][details_tags.get('user')])
            refs.append(details_arr[t][details_tags.get('ref')])
            
    return all_raw_data, filenames, data_lengths, file_users, refs    

########################################################################################################################
# Read in experiment details from a spreadsheet (specifcying filenames, user, reference electrode, length of experiment
def read_spreadsheet(spreadsheet, input_users):

    # intializations
    num_user_files = 0
    details_arr = []
    
    # read details spreadsheet csv file
    details = np.array(pd.read_csv(spreadsheet))
    
    # for each experiment (row) in spreadsheet
    for x in list(range(1,len(details[0:]))):
    
        # if user matches inputed user or all, then continue to read details for file
        for y in list(range(0,len(details[0]))):
            if (details[0][y] == 'user'):
                user = details[x][y]
        if (user == input_users):
            num_user_files = num_user_files + 1
        elif (input_users == 'all'):
            num_user_files = num_user_files + 1
        else:
            continue    
            
        # read in filename
        for y in list(range(0,len(details[0]))):
            if (details[0][y] == 'filename'):
                full_filename = details[x][y]   
                if '.hdf' in full_filename:
                    filename = full_filename.replace('.hdf', '')
                    file_extention = '.hdf'
                if '.mat' in full_filename: 
                    filename = full_filename.replace('.mat', '')
                    file_extention = '.mat'        
        
        # read in electrode used as reference    
        for y in list(range(0,len(details[0]))):
            if (details[0][y] == 'ref'):
                ref = int(details[x][y])
                    
        # read in length of experiment
            for y in list(range(0,len(details[0]))):
                if (details[0][y] == 'experiment_length'):          
                    experiment_length = int(details[x][y])

        details_arr.append([filename, file_extention, user, experiment_length, ref])
    details_tags = {"filename":0,"file extention":1, "user":2,"experiment length":3,"ref":4}   
    
    return details_arr, details_tags  
########################################################################################################################

    
########################################################################################################################
# Read in Ear EEG data (from a .mat or .hdf file!)
def get_ear_eeg(filename, file_extention, data_filepath, experiment_length, data_chs, fs):

    # read in data from hdf or mat file (hdf read in is NOT well tested, confirm it matches matlab before using!)
    full_filepath = data_filepath[:-1] + filename + file_extention
    if '.mat' in file_extention:
        # wand channels (.hdf and .mat formatting is different)
        wand_chs = np.array([30,26,22,18,14,10,6,2,12,9,3,1,61,57,53,49,45,41,37,33,52])
        # account for python 0 indexing (we call it 'channel 1', it's really index 0)
        ch_inxs = wand_chs[np.array(data_chs)-1]
        data_struct = scipy.io.loadmat(full_filepath)
        data_arr = data_struct.get('raw')
        crcs = data_struct.get('crc')
    elif '.hdf' in file_extention:
        # wand channels (.hdf and .mat formatting is different)
        wand_chs = np.array([31,27,23,19,15,11,7,3,12,10,4,2,62,58,54,50,46,42,38,34,53])
        # account for python 0 indexing (we call it 'channel 1', it's really index 0)
        ch_inxs = wand_chs[np.array(data_chs)-1]
        data_struct = h5py.File(full_filepath, "r")
        group_key=list(data_struct.keys())[0]
        data_group = data_struct[group_key]
        data_arr = data_group.get('dataTable')[()]
    else:
        print('Invalid filetype')
    
    # if experiment length is greater than data read in, reset experiment length
    if (data_arr.shape[0] < (experiment_length*fs)):
        len_dim = data_arr.shape[0]
        print('rewriting experiment length of ' + str(experiment_length) + ' to ' + str(math.floor(data_arr.shape[0]/fs)))
    else:
        len_dim = (experiment_length*fs)
    
    # initialize array for read-in data
    ch_dim = len(data_chs)
    sliced_data_arr = np.zeros((len_dim, ch_dim))    
    
    # slice data table and correct read out problems (non-zero values in first row of hdf file)
    if '.hdf' in file_extention:
        ch_count = 0  
        for x in wand_chs:
            if x in ch_inxs:
                #print(x)
                ind_count = 0
                for y in range(0, len_dim):   
                    if (data_arr[y][0][0] != 0):
                        sliced_data_arr[ind_count][ch_count] = sliced_data_arr[ind_count-1][ch_count]
                    elif(data_arr[y][0][x] != 0):
                        sliced_data_arr[ind_count][ch_count] = data_arr[y][0][x+1]               
                    ind_count = ind_count + 1
                ch_count = ch_count + 1
    
    # slice data table and correct read out problems (non-zero values in crcs)    
    elif '.mat' in file_extention:
        ch_count = 0  
        for x in wand_chs:
            if x in ch_inxs:
                ind_count = 0
                for y in range(0, len_dim):   
                    if (crcs[0][y] != 0) & (ind_count == 0):
                        sliced_data_arr[ind_count][ch_count] = sliced_data_arr[ind_count+1][ch_count]
                    elif(crcs[0][y] != 0):
                        sliced_data_arr[ind_count][ch_count] = sliced_data_arr[ind_count-1][ch_count]
                    elif(data_arr[y][x] != 0):
                        sliced_data_arr[ind_count][ch_count] = data_arr[y][x+1]                             
                    ind_count = ind_count + 1
                ch_count = ch_count + 1    
                
    # scale data and adjust for bits
    n_bits = 15
    full_scale = 0.1
    scale = full_scale / (2**n_bits - 1)
    for x in range(0,len(sliced_data_arr)):
        for y in range(0, len(sliced_data_arr[0])):
            if sliced_data_arr[x][y] > ((2**n_bits) - 1):
                sliced_data_arr[x][y] = sliced_data_arr[x][y] - 2**n_bits
    sliced_data_arr = (sliced_data_arr * scale) - full_scale/2 

    return sliced_data_arr 
########################################################################################################################