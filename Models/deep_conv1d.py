from tensorflow.keras.layers import Dense, Conv1D, Convolution1D, Conv2D, MaxPool1D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Input, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import math
import os

"""
    we train on the normal deep conv1d also with large kernel size 
    without batchnorm at the early layers
    with linear activation function
"""
def network(X_train, y_train, X_test, y_test):
    K.clear_session()
    im_shape = (X_train.shape[1], 1)
    inputs_cnn = Input(shape=(im_shape), name='inputs_cnn')
    #     conv1_1=Conv1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1 = Conv1D(32, (40), input_shape=im_shape, padding="same")(inputs_cnn)
    #     conv1_1=BatchNormalization()(conv1_1)
    pool1 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv1_1)
    #     conv2_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1 = Conv1D(32, (40), input_shape=im_shape, padding="same")(pool1)
    #     conv2_1=BatchNormalization()(conv2_1)
    pool2 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv2_1)
    #     conv3_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1 = Conv1D(32, (40), input_shape=im_shape, padding="same")(pool2)
    #     conv3_1=BatchNormalization()(conv3_1)
    pool3 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv3_1)
    conv4_1 = Conv1D(32, (40), input_shape=im_shape, padding="same")(pool3)
    #     conv3_1=BatchNormalization()(conv3_1)
    pool4 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv4_1)
    conv5_1 = Conv1D(32, (40), input_shape=im_shape, padding="same")(pool4)
    #     conv3_1=BatchNormalization()(conv3_1)
    pool5 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv5_1)
    conv6_1 = Conv1D(32, (33), input_shape=im_shape, padding="same")(pool5)
    conv6_1 = BatchNormalization()(conv6_1)
    pool6 = MaxPool1D(pool_size=(15), strides=(2), padding="same")(conv6_1)
    conv7_1 = Conv1D(32, (33), input_shape=im_shape, padding="same")(pool6)
    conv7_1 = BatchNormalization()(conv7_1)
    pool7 = MaxPool1D(pool_size=(10), strides=(2), padding="same")(conv7_1)
    conv8_1 = Conv1D(32, (25), input_shape=im_shape, padding="same")(pool7)
    conv8_1 = BatchNormalization()(conv8_1)
    pool8 = MaxPool1D(pool_size=(10), strides=(2), padding="same")(conv8_1)
    conv9_1 = Conv1D(32, (25), input_shape=im_shape, padding="same")(pool8)
    conv9_1 = BatchNormalization()(conv9_1)
    pool9 = MaxPool1D(pool_size=(10), strides=(1), padding="same")(conv9_1)
    conv10_1 = Conv1D(32, (25), input_shape=im_shape, padding="same")(pool9)
    #     conv3_1=BatchNormalization()(conv3_1)
    pool10 = MaxPool1D(pool_size=(10), strides=(1), padding="same")(conv10_1)
    flatten = Flatten()(pool10)
    #     dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(flatten)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)

    model = Model(inputs=inputs_cnn, outputs=main_output)
    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    abpath = os.path.abspath('Models')
    savepath = abpath + '/trained_models/model_linear_256.h5'
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ModelCheckpoint(filepath=savepath, monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, y_train, epochs=70, callbacks=callbacks, batch_size=32,
                        validation_data=(X_test, y_test))

    model.save(savepath)
    model.load_weights(savepath)
    return (model, history)