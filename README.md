# ECG Classification
This was a project for MLTS Course
The deep learning models are based on Tensorflow/Keras

## DATASET 
[here is the original dataset](https://physionet.org/content/mitdb/1.0.0/) \
Please download the dataset from the website.
There is also data description and annotation for the dataset. 

There is also a really similar dataset on [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat). 

The dataset is quite good for practicing deep learning. However, we have noticed that two-channel ECG recordings were obtained only from 47 subjects, although the data samples are a lot. \
It would be better that we can separate between the subjects, but the data we obtained were already mixed. There was a doubt on overfitting and data were not really generalized. \
Though, this was still a good training for Deep learning in Time series data.\

## Data Processing
First, we did data splitting in ```RawData_split.py``` to get the annotation corresponding to the data. \
in ```utils.py``` one can find how we prepare data. \
```split``` which splits the dataset into training set, validation set and test set. \
```upsample``` which is a function sklearn, trying to resample the different classes of data, due to the lack of some sample classes.\
Here was because the normal ECG samples are too more than the samples of other classes. 

```ecg2fig``` which enables one to plot the a single ecg sample into an image. 

```spectrogram``` which transfers the signal into spectrogram to have a frequency view of the samples.

## Training and Evaluation
in ```DataExplore_Train.ipynb``` 
The file completed how we preprocessed the data and trained on the models under the directory `/models`\
Our proposed models have similar architectures, which try to look for the important features in the samples. Therefore, our kernel sizes in the Convolutional layers are quite huges, which we thought that it should easiler to figure out where the important features are and get rid of noise. 

Our Accuracy on the validation set reached bestly 97.00%, we also provided Confusion Matrix. 




