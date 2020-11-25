import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from scipy.ndimage import gaussian_filter
from scipy import signal
import cv2


## plot the data classes as a circle to view the unbalance between the classes
def plot_num_of_classes(labels):
    plt.figure(figsize=(20, 10))
    my_circle = plt.Circle((0,0), 0.7, color="white")
    plt.pie(labels, labels= ['n', 'q', 'v', 's', 'f'], colors=
            ['red', 'green', 'blue', 'skyblue', 'orange'], autopct='%1.1f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

## split the data into train, validation, and test
## validation and test each class has 100 samples
## input_size of the sample in the dataset
def split(dataset, input_size=256):

    input_size = input_size + 1   # plus the label
    test_label = []
    validation_label = []

    validation = []
    test = []
    train_index = []
    num_train_index = 0
    ## seed the split algorithm
    np.random.seed(0)
    ## for validation and test dataset which has 100 samples for each class
    num_classes = 5
    for i in range(num_classes):
        ind = np.where(dataset[:, -1] == i)[0]  ## return index
        print('number of class :', len(ind))

        idx = np.random.permutation(ind)
        rest = idx[:200]
        train_idx = idx[200:]
        print('number of train data per class', len(train_idx))
        train_index.append(train_idx)
        #         train.append(ecgsignal[train_idx, :])
        validation.append(dataset[rest[:100], :-1])
        test.append(dataset[rest[100:], :-1])

        validation_label.append(dataset[rest[:100], -1])
        test_label.append(dataset[rest[100:], -1])
        #         train_label.append(target[train_idx]) ### training data needs to be noted the number of dataset
        num_train_index += len(train_index[i])
        print(len(train_index[i]))
    train = np.zeros((num_train_index, input_size))
    train_label = np.zeros((num_train_index, 1))

    train[0:len(train_index[0]), :] = dataset[train_index[0], :]
    culm = len(train_index[0]) + len(train_index[1])
    train[len(train_index[0]):culm, :] = dataset[train_index[1], :]
    train[culm:(culm + len(train_index[2])), :] = dataset[train_index[2], :]
    culm += len(train_index[2])
    train[culm:(culm + len(train_index[3])), :] = dataset[train_index[3], :]
    culm += len(train_index[3])
    train[culm:(culm + len(train_index[4])), :] = dataset[train_index[4], :]
    culm = len(train_index[0]) + len(train_index[1])
    train_label[0:len(train_index[0]), 0] = dataset[train_index[0], -1]
    train_label[len(train_index[0]):culm, 0] = dataset[train_index[1], -1]
    train_label[culm:(culm + len(train_index[2])), 0] = dataset[train_index[2], -1]
    culm += len(train_index[2])
    train_label[culm:(culm + len(train_index[3])), 0] = dataset[train_index[3], -1]
    culm += len(train_index[3])
    train_label[culm:(culm + len(train_index[4])), 0] = dataset[train_index[4], -1]

    validation = np.concatenate(validation, 0)
    test = np.concatenate(test, 0)
    test_label = np.concatenate(test_label, 0)
    validation_label = np.concatenate(validation_label, 0)
    return (train, train_label), (validation, validation_label), (test, test_label)

## upsample function
## due to the unbalance of the classes
### according to https://elitedatascience.com/imbalanced-classes
def upsample(train, upsample_size=10000):
    df_1 = train[train[:, -1] == 1]
    df_2 = train[train[:, -1] == 2]
    df_3 = train[train[:, -1] == 3]
    df_4 = train[train[:, -1] == 4]
    idxs = np.random.choice(train[train[:, -1] == 0].shape[0], upsample_size, replace=False)
    df_0 = train[idxs]
    #df_0 = (train[train_label == 0]).sample(n=20000, random_state=42)
    df_1 = pd.DataFrame(df_1)
    df_2 = pd.DataFrame(df_2)
    df_3 = pd.DataFrame(df_3)
    df_4 = pd.DataFrame(df_4)
    df_0 = pd.DataFrame(df_0)
    df_1_upsample = resample(df_1, replace=True, n_samples=upsample_size, random_state=123)
    df_2_upsample = resample(df_2, replace=True, n_samples=upsample_size, random_state=124)
    df_3_upsample = resample(df_3, replace=True, n_samples=upsample_size, random_state=125)
    df_4_upsample = resample(df_4, replace=True, n_samples=upsample_size, random_state=126)
    train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])
    return train_df

def ecg2fig(data):
    data = data.reshape(data.shape[0], data.shape[1])
    imgs = np.zeros((data.shape[0], 128, 128), dtype='uint8')
    for i in range(data.shape[0]):
        fig = plt.figure(frameon=False)
        plt.plot(data[i]) 
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(img, (128, 128), interpolation = cv2.INTER_LANCZOS4)
        imgs[i] = im_gray
        plt.close(fig)
    return imgs

def spectrogram(data):
    data = data.reshape(data.shape[0], data.shape[1])
#     img = np.zeros((data.shape[0], 129, 1), dtype='uint8')
    fs = 360
    f, t, Sxx = signal.spectrogram(data, fs=fs)
    print(Sxx.shape)
    Sx = Sxx[:, :, 0]
    print(Sx.shape)
    return Sx, Sxx
    