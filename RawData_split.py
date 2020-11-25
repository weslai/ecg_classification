# import require packages
from __future__ import division, print_function
import pandas as pd
import wfdb
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from scipy.signal import find_peaks


## you need to store it after you finish this function
def data_prepare(raw_datapath, save_path, sample_size=256):
    """
    split the raw data into dataset
    :param raw_datapath: the path to the raw data from mit-bih
    :param save_path: the path to save the ecg and labels (ex: "Data")
    :param data_size: the size of each sample in dataset (default:256)
    :return: ecgs and labels
    """
    ## data path
    data_path = raw_datapath
    ## sample size
    data_size = sample_size

    ## data lists
    pts = ['100', '104', '108', '113', '117', '122', '201', '207', '212', '217', '222', '231',
           '101', '105', '109', '114', '118', '123', '202', '208', '213', '219', '223', '232',
           '102', '106', '111', '115', '119', '124', '203', '209', '214', '220', '228', '233',
           '103', '107', '112', '116', '121', '200', '205', '210', '215', '221', '230', '234']

    ## map the ~19 classes to 5 classes
    ## according to the paper https://arxiv.org/pdf/1805.00794.pdf
    mapping = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'B': 0,  # N = 0
               'A': 1, 'a': 1, 'J': 1, 'S': 1,  # S = 1
               'V': 2, 'E': 2, 'r': 2, 'n': 2,  # V = 2
               'F': 3,  # F = 3
               '/': 4, 'f': 4, 'Q': 4, '?': 4}  # Q = 4
    ignore = ['+', '!', '[', ']', 'x', '~', '|', '"']

    ## we split the each set of the data into size 256( which we can see the ecg pulse, just one pulse)
    def dataSaver(dataset=pts, data_size=data_size):
        input_size = data_size  ## default

        def dataprocess():
            ecg = np.zeros((1, input_size))
            label = np.zeros((1, 1))
            for num in tqdm(dataset):
                print(num, 'now')
                idx = 0  ## count for the matrixes
                record = wfdb.rdrecord(data_path + num, smooth_frames=True)

                ## normalize the data ecg
                signals0 = np.nan_to_num(record.p_signal[:, 0])
                # signals1 = np.nan_to_num(record.p_signal[:, 1])
                min_max_scaler = preprocessing.MinMaxScaler()
                signals0 = min_max_scaler.fit_transform(signals0.reshape(-1, 1))
                # signals1 = min_max_scaler.fit_transform(signals1.reshape(-1, 1))
                signals0 = signals0.reshape(-1)
                # signals1 = signals1.reshape(-1)

                ## find peaks # R-peaks
                ## we only use the channel 0
                peaks, _ = find_peaks(signals0, distance=150)

                X = np.zeros((len(peaks), input_size))
                Y = np.zeros((len(peaks), 1))

                # skip a first peak to have enough range of the sample
                # in the for loop, we look for the annotation
                for peak in tqdm(peaks[1:-1]):
                    start, end = peak - input_size // 2, peak + input_size // 2
                    start = max([0, start])
                    end = min([len(signals0), end])
                    ann = wfdb.rdann(data_path + num, extension='atr', sampfrom=start, sampto=end,
                                     return_label_elements=['symbol'])
                    symbol = ann.symbol
                    count = 0
                    if len(symbol) != 1:
                        for sym in symbol:
                            if sym in ignore:
                                count += 1
                                continue
                            elif sym == 'N':
                                continue
                            else:
                                symbol = sym
                                break
                    if count > 0 and len(symbol) > 1:
                        symbol = '+'
                    elif len(symbol) > 1:
                        symbol = 'N'
                    elif len(symbol) == 0:
                        symbol = '+'
                    assert len(symbol) <= 1, "the symbol is not only one.{} len".format(len(symbol))

                    if len(symbol) == 1:
                        for ss in symbol:
                            if ss in ignore:
                                continue
                            else:
                                Y[idx, 0] = mapping[ss]
                                sig = signals0[start:end]
                                X[idx, :len(sig)] = sig
                                idx += 1
                ecg = np.concatenate((ecg, X), axis=0)
                label = np.concatenate((label, Y), axis=0)
            ecg = ecg[1:, :]
            label = label[1:, :]
            ecg = pd.DataFrame(ecg)
            label = pd.DataFrame(label)

            return ecg, label
        ecg, label = dataprocess()
        return ecg, label

    ecg, label = dataSaver(pts)
    ecg_path = save_path + "/ecg_signal_{}.csv".format(data_size)
    label_path = save_path + "/label_{}.csv".format(data_size)
    ecg.to_csv(ecg_path, index=None, header=None)
    label.to_csv(label_path, index=None, header=None)
    return ecg, label

# if __name__ == "__main__":
#     ecg, label = data_prepare("mit-bih-arrhythmia-database-1.0.0/", "Data")
