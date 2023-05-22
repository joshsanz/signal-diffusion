import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_in_labels(filenames, data_lengths, label_filepath, plot_labels_enable):

    # for each ear eeg file, read in cooresponding label
    for x in range(0,len(filenames)):

        # Arya add start
        if x > 21:
            break
        # Arya end
    
        print('reading ' + filenames[x] + ' labels')
        # read label file
        labels = np.array(pd.read_csv(label_filepath + str(filenames[x]) + '_labels.csv'))
        
        # set experiment length to match eeg data
        labels = np.array(labels[:data_lengths[x]])
        
        # plot labels
        if plot_labels_enable:
            # plot trial labels
            plt.plot(list(range(0,len(labels))),labels)
            plt.title(filenames[x] + ' labels')
            plt.show()

        #Arya label fix start
        #while len(labels) % 10 != 0:
        #    labels = np.append(labels, [labels[-1]])

        labels = np.append(labels, labels[-1])
        labels = np.array(labels)
        labels = np.squeeze(labels)



        #Arya label fix end



        
        # concatenate all labels
        if(x==0):
            # all_labels = labels

            # arya start
            all_labels = []
            all_labels.append(labels)
            #arya end
            
        else:
            # all_labels = np.concatenate([all_labels,labels])
            all_labels.append(labels)
    
    return all_labels