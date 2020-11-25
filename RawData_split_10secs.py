## load the data from mit-bih-arrhythmia
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

## split the raw data into 10 secs for each sample in dataset
def data_prepare(raw_datapath, save_path):
    """
    split the raw data into dataset (each sample for 10 secs)
    :param raw_datapath: the path to the raw data from mit-bih
    :param save_path: the path to save the ecg and labels (ex: "Data")
    :return: ecgs and labels
    """
    ## data path
    data_path = raw_datapath

    ## data lists

    # pts = ['100', '104', '108', '113', '117', '122', '201', '207', '212', '217', '222', '231',
    #        '101', '105', '109', '114', '118', '123', '202', '208', '213', '219', '223', '232',
    #        '102', '106', '111', '115', '119', '124', '203', '209', '214', '220', '228', '233',
    #        '103', '107', '112', '116', '121', '200', '205', '210', '215', '221', '230', '234']
    pts = ["100"]
    ## map the ~19 classes to 5 classes
    ## according to the paper https://arxiv.org/pdf/1805.00794.pdf
    mapping = {'N':0, 'L':0, 'R':0, 'e':0, 'j':0, 'B':0, # N = 0
               'A':1, 'a':1, 'J':1, 'S':1, # S = 1
               'V':2, 'E':2, 'r':2, 'n':2, # V = 2
               'F':3, # F = 3
               '/':4, 'f':4, 'Q':4, '?':4} # Q = 4
    ignore = ['+', '!', '[', ']', 'x', '~', '|', '"']

    secs = 5
    sample_rates = 360
    num_cols = 2 * secs * sample_rates
    ecg = np.zeros((1, num_cols))
    label = np.zeros((1, 1))

    for file in pts:
        # first load the simple file
        # load the ecg signal file
        x = wfdb.rdsamp(data_path + file)
        ecg_signal = x[0][:, 0]

        # vertify frequency ist 360
        # assert x[1]['fs'] == 360, 'sample freq is not 360'

        # load the annotations (in the symbol is the classes)
        annotation = wfdb.rdann(data_path + file, 'atr')
        atr_symbol = annotation.symbol
        atr_samples = annotation.sample

        classes = []
        new_samples = []
        idx = 0
        for i in atr_symbol:
            if i in ignore:
                idx += 1
                continue
            classes.append(mapping[i])
            new_samples.append(atr_samples[idx])
            idx += 1

        # make a df
        df_ann = pd.DataFrame({"atr_symbol" : classes,
                               "atr_samples" : new_samples})
        num_rows = len(df_ann)

        X = np.zeros((num_rows, num_cols))
        Y = np.zeros((num_rows, 1))
        max_row = 0

        for i, j in zip(new_samples, classes):
            left = max([0, (i - secs * sample_rates)])
            right = min([len(ecg_signal), (i + secs * sample_rates)])
            signal = ecg_signal[left:right]

            if len(signal) == num_cols:
                X[max_row, :] = signal
                Y[max_row, :] = j
                max_row += 1
            else:
                X[max_row, :len(signal)] = signal
                Y[max_row, :] = j
                max_row += 1
        # merge all the signal and label in two arrays
        ecg = np.concatenate((ecg, X), axis= 0)
        label = np.concatenate((label, Y), axis= 0)

    # data normalization between 0 and 1
    data_min = np.min(ecg, axis= 1).reshape(-1, 1)
    data_max = np.max(ecg, axis = 1).reshape(-1, 1)
    ecg = (ecg - data_min)/(data_max - data_min)

    # not using the first row
    ecg = ecg[1:, :]
    label = label[1:, :]
    ecg = pd.DataFrame(ecg)
    label = pd.DataFrame(label)
    ecg_path = save_path + "/ecg_signal.csv"
    label_path = save_path + "/label.csv"
    ecg.to_csv(ecg_path, index= None, header= None)
    label.to_csv(label_path, index= None, header= None)
    return ecg, label

# if __name__ == "__main__":
#     ecg, label = data_prepare("mit-bih-arrhythmia-database-1.0.0/", "Data")

## plot the ecg
# time = np.arange(len(ecg_signal))
# plt.figure()
# plt.plot(time[0:3600], ecg_signal[0:3600], '-')
# plt.show()
