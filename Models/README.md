## Deep Learning Models

## Architectures
here we tried with 1D and 2D Convolutional Neural Networks. However, the 2D CNN was not suitable for ECG signals, as it was too time-consuming and we would also need to convert the ECG samples to image samples. 

In the directory ```trained_models```, one can find the trained weights with model architectures. 

In the file ```utils.py```, we provide the methods to evaluate our models and compute the confusion matrixs. 
