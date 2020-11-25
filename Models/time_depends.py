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
## we do this model with high kernel size to relate the time dependent signal like ecg with each other
## model name save as model_kernel40.h5
def time_depends(x_train, y_train, x_val, y_val):
    K.clear_session()
    im_shape = (x_train.shape[1], 1)
    inp = Input(shape=(im_shape), name='inputs_cnn')
    C = Conv1D(filters=32, kernel_size=40, strides=1)(inp)

    C11 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(A11)
    S11 = Add()([C12, C])
    A12 = Activation("relu")(S11)
    M11 = MaxPool1D(pool_size=15, strides=2)(A12)

    C21 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(M11)
    A21 = Activation("relu")(C21)
    C22 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(A21)
    S21 = Add()([C22, M11])
    A22 = Activation("relu")(S11)
    M21 = MaxPool1D(pool_size=15, strides=2)(A22)

    C31 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(M21)
    A31 = Activation("relu")(C31)
    C32 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(A31)
    S31 = Add()([C32, M21])
    A32 = Activation("relu")(S31)
    M31 = MaxPool1D(pool_size=10, strides=2)(A32)

    C41 = Conv1D(filters=32, kernel_size=24, strides=1, padding='same')(M31)
    A41 = Activation("relu")(C41)
    C42 = Conv1D(filters=32, kernel_size=24, strides=1, padding='same')(A41)
    S41 = Add()([C42, M31])
    A42 = Activation("relu")(S41)
    M41 = MaxPool1D(pool_size=10, strides=2)(A42)

    C51 = Conv1D(filters=32, kernel_size=24, strides=1, padding='same')(M41)
    A51 = Activation("relu")(C51)
    C52 = Conv1D(filters=32, kernel_size=24, strides=1, padding='same')(A51)
    S51 = Add()([C52, M41])
    A52 = Activation("relu")(S51)
    M51 = MaxPool1D(pool_size=10, strides=1)(A52)

    C61 = Conv1D(filters=32, kernel_size=12, strides=1, padding='same')(M51)
    A61 = Activation("relu")(C61)
    C62 = Conv1D(filters=32, kernel_size=12, strides=1, padding='same')(A61)
    S61 = Add()([C62, M51])
    A62 = Activation("relu")(S61)
    M61 = MaxPool1D(pool_size=5, strides=1)(A62)

    C71 = Conv1D(filters=32, kernel_size=12, strides=1, padding='same')(M61)
    A71 = Activation("relu")(C71)
    C72 = Conv1D(filters=32, kernel_size=12, strides=1, padding='same')(A71)
    S71 = Add()([C72, M61])
    A72 = Activation("relu")(S71)
    M71 = MaxPool1D(pool_size=5, strides=1)(A72)

#     C81 = Conv1D(filters=32, kernel_size=20, strides=1, padding='same')(M71)
#     A81 = Activation("relu")(C81)
#     C82 = Conv1D(filters=32, kernel_size=20, strides=1, padding='same')(A81)
#     S81 = Add()([C82, M71])
#     A82 = Activation("relu")(S81)
#     M81 = MaxPool1D(pool_size=5, strides=2)(A82)
    F1 = Flatten()(M71)

#     D1 = Dense(32)(F1)
#     A6 = Activation("relu")(D1)
    #     D2 = Dense(32)(A6)
    D3 = Dense(5)(F1)
    A7 = Softmax()(D3)

    model = Model(inputs=inp, outputs=A7)

    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # def exp_decay(epoch):
        # initial_lrate = 0.001
        # k = 0.75
        # n_obs = train.shape[0]
        # batch_size = 32
        # t = n_obs // (10000 * batch_size)  # every epoch we do n_obs/batch_size iteration
        # lrate = initial_lrate * math.exp(-k * t)
        # return lrate

    # lrate = LearningRateScheduler(exp_decay)
    # callbacks = [lrate, ModelCheckpoint(filepath='../trained_models/model_kernel40.h5', save_weights_only=True,
    #                                     verbose=1), EarlyStopping(monitor='val_loss', patience=12)]
    abpath = os.path.abspath('Models')
    savepath = abpath + '/trained_models/model_kernel40.h5'
    callbacks = [EarlyStopping(monitor='val_loss', patience=18),
                 ModelCheckpoint(filepath=savepath, monitor='val_loss', save_best_only=True)]

    history = model.fit(x_train, y_train, epochs=55, callbacks=callbacks, batch_size=64, validation_data=(x_val, y_val))
    model.save(savepath)
    model.load_weights(savepath)

    return (model, history)