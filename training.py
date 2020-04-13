'''
#######################################################################
Script to train models for channel estimation with DL
#######################################################################

Authors:

Wesin Ribeiro
Yuichi Takeda
Aldebaro Klautau
LASSE/PPGEE/UFPA
2019

#######################################################################
conventions:
#######################################################################

y_ is the input of the model and represents signal received by antennas
h_ is the output of the model and represents the channel

'''
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Embedding,
    Input,
    Reshape,
    Dense,
    Lambda,
    Dropout,
    Flatten,
    MaxPooling2D,
    MaxPooling1D,
    AveragePooling1D,
    AveragePooling2D,
    Conv2D,
    Conv1D,
    GRU,
    LSTM,
    BatchNormalization,
    Add,
)

from keras import regularizers
from keras.layers.merge import concatenate

from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator

from keras.callbacks import TensorBoard

from matplotlib import pyplot as plt
import argparse
from keras.models import model_from_json
from keras.constraints import max_norm
import numpy.linalg as la
from keras import backend as K
import os
import shutil
import sys
import json
from datetime import datetime
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

##### hack my life ######################
#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = K.tf.Session(config=config)
#########################################


logdir = "./mimo8x8"
# save this script
os.makedirs(logdir, exist_ok=True)
ini = sys.argv[0]
shutil.copyfile(sys.argv[0], os.path.join(logdir, os.path.basename(sys.argv[0])))
print("Copied:", sys.argv[0], "to", os.path.join(logdir, os.path.basename(sys.argv[0])))

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

def normalize_arr(a):
    a_std = (a - np.min(a)) / (np.max(a) - np.min(a))
    return a_std * (1 - (-1)) + (-1) # X_std * (max - min) + min

def normalize_output(x):
    print("x1=", x)
    norm = np.sum(x ** 2)
    # x = x / np.sqrt(norm)
    x = x / norm

    print("x2=", x)
    print(norm)
    return x


# training parameters
epochs = 100
batch_size = 5

frequency = "60GHz"
quantization = "1-bit"
#scaler = MinMaxScaler()
scaler = StandardScaler()

global Nt
global Nr
Nr = 8
Nt = 8


# load dataset
print('Loading data...')
mat = loadmat("E:\\wesin\\Doutorado\\channel-estimation\\datasets\\{}\mimo8x8_256samples_{}".format(quantization,frequency))
inputs, outputs = [mat[key] for key in ["inputs", "outputs"]]
#tmp = outputs.reshape((-1,2 * Nt * Nr))
#outputs = scaler.fit_transform(tmp).reshape((-1,2 * Nt,Nr))
y_train,y_test,h_train,h_test = train_test_split(inputs,outputs,shuffle=True, test_size=0.1)




numSamplesPerExample = 256
global H_normalization_factor
H_normalization_factor = np.sqrt(Nr * Nt)

global K_H_normalization_factor
K_H_normalization_factor = K.variable(H_normalization_factor, dtype=np.float32)

input_shape = (numSamplesPerExample, 2 * (Nr))
output_dim = (2 * Nr, Nt)

numInputs = np.prod(input_shape)
numOutputs = np.prod(output_dim)
print(numInputs, " ", numOutputs)

'''
# normalizing values
y_train[y_train < 0] = 0
y_test[y_test < 0] = 0

h_train = h_train.reshape((-1, numOutputs))
h_test = h_test.reshape((-1, numOutputs))

print('shape: ', h_train.shape, h_test.shape)

scaler.fit(h_train)


h_train = scaler.transform(h_train)
h_test = scaler.transform(h_test)

h_train = h_train.reshape((-1,2*Nr, Nt))
h_test = h_test.reshape((-1,2*Nr, Nt))
'''

# sanity check
'''
print(input_shape, output_dim)
print(inputs.shape, y_train.shape, y_test.shape)
print(outputs.shape, h_train.shape, h_test.shape)
'''

# create model

def create_dense():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # K_H_normalization_factor = K.variable(H_normalization_factor,dtype=np.float32)

    N = 150
    model = Sequential()

    model.add(Dense(N, activation="tanh", input_shape=input_shape))
    model.add(Flatten())
   
    model.add(Dense(N, activation="tanh"))

    model.add(Dense(numOutputs, activation="linear"))   
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


def create_res():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # residual net using Kera's functional API (not using Sequential)
    N = 10
    inputs = Input(input_shape)
    x = Dense(1 * N, activation="tanh")(inputs)
    y = Flatten()(x)
    # yn = BatchNormalization()(y)
    y2 = Dense(N, activation="tanh")(y)
    # d2 = Dropout(0.8)(y2)
    # yn2 = BatchNormalization()(d2)
    # predictions = Dense(10, activation='softmax')(x)
    y2res = Dense(1, activation="tanh")(y2)
    z = keras.layers.add([y, y2res])
    
    
    
    
    y3 = Dense(numOutputs, activation="linear")(z)
    y4 = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(y3)
    # d3 = Dropout(0.8)(y3)
    # y4 = Embedding()(y3)
    y5 = Reshape(output_dim)(y4)
    # z = keras.layers.add([x, y])
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=y5)
    return model


def create_refinet():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)

    def refinet(y):
        shortcut = y
        x = Conv1D(128,3, padding = 'same', activation = 'tanh')(shortcut)
        x = Conv1D(128,3, padding = 'same', activation = 'tanh')(x)
        #x = Conv1D(128,6, activation = 'tanh', padding='same')(x)

        x = Add()([shortcut,x])

        return x



    inputs = Input(input_shape)
    #x = Input(input_shape)

    #x = Reshape((2 * Nr, numSamplesPerExample))(inputs)
    x = Conv1D(128,3, padding = 'same', activation = 'tanh')(inputs)

    for i in range(3):
        x = refinet(x)
        x = Dropout(0.3)(x)
        #x = AveragePooling1D(2)(x)
    
    #x = AveragePooling1D(2)(x)
    y2 = Flatten()(x)
    #y2 = Dense(150, activation="tanh")(y2)
    y3 = Dense(numOutputs, activation="linear")(y2)
    y4 = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(y3)
    y5 = Reshape(output_dim)(y4)
    model = Model(inputs=inputs, outputs=y5)
    
    return model

def create_res2():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # residual net using Kera's functional API (not using Sequential)
    N = 10
    inputs = Input(input_shape)
    x = Dense(2 * N, activation="relu")(inputs)
    y = Flatten()(x)
    # yn = BatchNormalization()(y)
    d1 = Dropout(0.2)(y)
    y2 = Dense(2 * N, activation="relu")(d1)
    d2 = Dropout(0.2)(y2)
    # yn2 = BatchNormalization()(d2)
    # predictions = Dense(10, activation='softmax')(x)
    # y2res = Dense(1, activation='relu')(d2)
    z = keras.layers.concatenate([d1, d2])

    y3 = Dense(numOutputs, activation="linear")(z)
    y4 = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(y3)
    # d3 = Dropout(0.8)(y3)
    # y4 = Embedding()(y3)
    y5 = Reshape(output_dim)(y4)
    # z = keras.layers.add([x, y])
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=y5)
    return model


def create_res3():
    global Nt
    global Nr
    norm_factor = np.sqrt(Nr * Nt)

    N = 40
    inputs = Input(input_shape)

    f = Flatten()(inputs)

    d1 = Dense(N, activation="relu")(f)

    conc1 = keras.layers.concatenate([f, d1])

    d2 = Dense(N, activation="relu")(conc1)

    conc2 = keras.layers.concatenate([d1, d2])

    d3 = Dense(N, activation="relu")(conc2)

    conc3 = keras.layers.concatenate([d2, d3])

    d_out = Dense(numOutputs, activation="linear")(conc3)

    lambda1 = Lambda(lambda x: norm_factor * K.l2_normalize(x, axis=-1))(d_out)

    reshap1 = Reshape(output_dim)(lambda1)

    model = Model(inputs=inputs, outputs=reshap1)
    return model

def create_conv():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    model = Sequential()
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Conv1D(64,4,activation="tanh", padding = "same", input_shape = input_shape ))
    #model.add(AveragePooling1D(2))
    model.add(Conv1D(128,4,activation="tanh", padding = "same" ))
    model.add(AveragePooling1D(4))
 
    model.add(Flatten())
    model.add(Dense(128, activation="tanh"))
    model.add(Dense(256, activation="tanh"))
    model.add(Dense(512, activation="tanh"))

    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model

def create_conv2():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Conv1D(64,3,activation="tanh", padding = "same", input_shape=input_shape ))
    model.add(Conv1D(64,3,activation="tanh", padding = "same"))
    model.add(Dropout(0.3))
 
 
    model.add(Flatten())
    #model.add(Dense(150, activation="tanh"))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model

def create_conv3():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Reshape((2 * Nr, numSamplesPerExample, 1), input_shape=input_shape))
    model.add(Conv2D(64,(3,3),activation="relu", padding = "same" ))
    model.add(Conv2D(64,(3,3),activation="relu", padding = "same"))
    #model.add(AveragePooling1D(2))
    model.add(Dropout(0.3))

   
    model.add(Flatten())
    #model.add(Dense(150, activation="tanh"))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model

def create_denoising():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Conv1D(64,3,activation="tanh", padding = "same", input_shape=input_shape ))
    model.add(Conv1D(32,3,activation="tanh", padding = "same"))
    model.add(Conv1D(8,3,activation="tanh", padding = "same"))
    model.add(Conv1D(32,3,activation="tanh", padding = "same"))
    model.add(Conv1D(64,3,activation="tanh", padding = "same"))
    model.add(Dropout(0.3))
 
 
    model.add(Flatten())
    #model.add(Dense(150, activation="tanh"))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


# define baseline model, avoid relu activation given that we have negative numbers
def baseline_model():
    # return create_lstm_stateful()
    # return create_conv1D()
    # return create_dense()
    # return create_conv()
    return create_conv2()
    # return create_res()
    # return create_res2()
    # return create_res3()
    # return create_refinet()
    # return create_denoising()
    # return create_attention()


# training model

weights_filename = "model_{}_weights_channel_estimation".format(1)
#model_filepath = "models/{}/model_refinet_256samples_{}.h5".format(quantization,frequency)
model_filepath = "models/model-china-conv.h5"
model = baseline_model()

model.compile(loss="mse", optimizer="adam", metrics=['mean_absolute_error'])
print(model.summary())

stateful = False
if stateful:
    for i in range(10):
        print('Epoch', i + 1, '/', '10')
        model.fit(
            y_train,
            h_train,
            shuffle = False,
            batch_size = 1,
            validation_split = 0.2,
            epochs= 1,
            verbose=1,    
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-7,
                    patience=5,
                    # restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=model_filepath,
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    min_delta=1e-7,
                    patience=2,
                    cooldown=5,
                    verbose=1,
                    min_lr=1e-6,
                ),
            ],
        )
        model.reset_states()

    # testing model

    predictedOutput = model.predict(y_test)
    error = h_test - predictedOutput
    mseTest = np.mean(error[:] ** 2)
    print("overall MSE = ", mseTest)
    mean_nmse = mseTest / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)


else:
    history = model.fit(
        y_train,
        h_train,
        shuffle = True,
        validation_split = 0.2,
        epochs= epochs,
        verbose=1,    
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-7,
                patience=5,
                # restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_filepath,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                min_delta=1e-7,
                patience=2,
                cooldown=5,
                verbose=1,
                min_lr=1e-6,
            ),
        ],
    )

    # testing model

    mse, mae = model.evaluate(y_test, h_test,
                                batch_size=batch_size)
    print('Test mse:', mse)
    print('Test mae:', mae)
    mean_nmse = mse / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)