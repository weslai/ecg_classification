from tensorflow.keras.layers import Dense, Conv1D, Convolution1D, Conv2D, MaxPool1D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Input, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Sequential
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import os

def model_2d(x_train, y_train, x_val, y_val):
    K.clear_session()
    batch_size = 64
    number_of_classes = 5
    n = x_train.shape[1]
    m = x_train.shape[2]
    c = x_train.shape[3]
    image_size = (n, m, c)
#     model = Sequential()
    #model.load_weights('my_model_weights.h5')
#     inp = Input(shape=(im_shape), name='inputs_cnn')
    inp = Input(shape=image_size, name='inputs_cnn')
    C = Conv2D(32, (7, 7), activation='relu', strides=1)(inp)
    B = BatchNormalization()(C)
    #32 conv
    C1 = Conv2D(32, (5, 5), activation='relu', input_shape=image_size, padding='same')(B)
#     B1 = BatchNormalization()(C1)
    C2 = Conv2D(32, (5, 5), activation='relu', padding='same')(C1)
    B2 = BatchNormalization()(C2)
    CC = Conv2D(32, (1, 1), activation='relu', padding='same')(B)
    S1 = Add()([B2, CC])
    M1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(S1)
    # model.add(Dropout(rate=0.1))
    #128 conv
    C3 = Conv2D(32, (3, 3), activation='relu', padding='same')(M1)
#     B3 = BatchNormalization()(C3)
    C4 = Conv2D(32, (3, 3), activation='relu', padding='same')(C3)
    B4 = BatchNormalization()(C4)
    CC = Conv2D(32, (1, 1), activation='relu', padding='same')(M1)
    S2 = Add()([B4, CC])
    M2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(S2)
    # model.add(Dropout(rate=0.1))
    # #256 conv
    C5 = Conv2D(32, (3, 3), activation='relu', padding='same')(M2)
#     B5 = BatchNormalization()(C5)
    C6 = Conv2D(32, (3, 3), activation='relu', padding='same')(C5)
    B6 = BatchNormalization()(C6)
    CC = Conv2D(32, (1, 1), activation='relu', padding='same')(M2)
    S3 = Add()([B6, CC])
    M3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(S3)
    
    C7 = Conv2D(32, (3, 3), activation='relu', padding='same')(M3)
#     B7 = BatchNormalization()(C7)
    C8 = Conv2D(32, (3, 3), activation='relu', padding='same')(C7)
    B8 = BatchNormalization()(C8)
    CC = Conv2D(32, (1, 1), activation='relu', padding='same')(M3)
    S4 = Add()([B8, CC])
    M4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(S4)
    # model.add(Dropout(rate=0.1))
    
#     C9 = Conv2D(128, (3, 3), activation='relu', padding='same')(M4)
#     B9 = BatchNormalization()(C9)
#     C10 = Conv2D(128, (3, 3), activation='relu', padding='same')(B9)
#     B10 = BatchNormalization()(C10)
#     CC = Conv2D(128, (1, 1), activation='relu', padding='same')(M4)
#     S5 = Add()([B10, CC])
#     M5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(S5)
    # model.add(Dropout(rate=0.2))

    #Dense part
    F1 = Flatten()(M4)
    D1 = Dense(32, activation='relu')(F1)
    D1 = Dense(number_of_classes, activation='softmax')(D1)
    model = Model(inputs=inp, outputs=D1)
    model.summary()
    
    train_image_generator = ImageDataGenerator(rescale=1./255)
                                               #featurewise_center=True)
#                                                featurewise_std_normalization=True,
#                                                rotation_range=20,
#                                                width_shift_range=0.2,
#                                                height_shift_range=0.2,
#                                                horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1./255)
    train_image_generator.fit(x_train)
    val_image_generator.fit(x_val)
    train_img_gen = train_image_generator.flow(x_train, y_train, 
                                               batch_size=batch_size)
    val_img_gen = val_image_generator.flow(x_val, y_val)
                                               

    abpath = os.path.abspath('Models')
    savepath = abpath + '/trained_models/twod_weights.h5'

    #Callbacks and accuracy calculation
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=8, verbose=1, mode='auto')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#     rmsprop = RMSprop(lr=0.001, rho=0.9, momentum=0.0, epsilon=1e-07)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    checkpointer = [early_stopping, 
                    ModelCheckpoint(filepath=savepath, monitor='val_loss', save_weights_only=False, period=1, verbose=1, save_best_only=False)]
    history = model.fit_generator(generator=train_img_gen, 
                                  validation_data=val_img_gen,
                                  epochs=25, 
                                  callbacks=checkpointer)
#     history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size, 
#                         validation_data=(x_val, y_val),callbacks=checkpointer)# shuffle=False, callbacks=[checkpointer])
    model.save(savepath)
    model.load_weights(savepath)
    return (model, history)