'''
#######################################################################
Script to test models for channel estimation with DL
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
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding,
    Input,
    Reshape,
    Dense,
    Lambda,
    Dropout,
    Flatten,
    MaxPooling2D,
    Conv2D,
    BatchNormalization,
)
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

frequency = "60GHz"
bits = "1-bit"


#logdir = "./mimo8x8/{}".format(quantization)
logdir = f"./mimo8x8/{bits}"
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


# Parameters
global Nt
global Nr
Nt = 32  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 8  # num of Tx antennas
# the sample is a measurement of Y values, and their collection composes an example. The channel estimation
min_randomized_snr_db = -1
max_randomized_snr_db = 1

# must be done per example, each one having a matrix of Nr x numSamplesPerExample of complex numbers
numSamplesPerExample = 256  # number of channel uses, input and output pairs
# if wants a gradient calculated many times with same channel
numExamplesWithFixedChannel = 1
numSamplesPerFixedChannel = (
    numExamplesWithFixedChannel * numSamplesPerExample
)  # coherence time
# obs: it may make sense to have the batch size equals the coherence time
batch_size = 1  # numExamplesWithFixedChannel

num_test_examples = 2000  # for evaluating in the end, after training
# get small number to avoid slowing down the simulation, test in the end
num_validation_examples = 2000
num_training_examples = 10000
file = "channels_rosslyn_60Ghz_Nr8Nt32_mobile_s004.mat"
#file = "channel_china60Ghz.mat"
method = "manual"
training_generator = RandomChannelMimoDataGenerator(
    batch_size=batch_size,
    Nr=Nr,
    Nt=Nt,
    # num_clusters=num_clusters,
    numSamplesPerFixedChannel=numSamplesPerFixedChannel,
    # numSamplesPerExample=numSamplesPerExample, SNRdB=SNRdB,
    numSamplesPerExample=numSamplesPerExample,
    # method='random')
    method=method,
    file = file
)
training_generator.randomize_SNR = False



#print(training_generator.method, training_generator.randomize_SNR)


# model_filepath = "models/{}/model_dense_256samples_{}.h5".format(quantization,frequency)
model_filepath = f"models/{bits}/model-beijing.h5"
model = keras.models.load_model(model_filepath)
SNRdB_values = np.arange(-21, 22, 3)
#training_generator.method = "manual"
print(model.summary())


all_nmse_db_average = np.zeros((SNRdB_values.shape))
all_nmse_db_min = np.zeros((SNRdB_values.shape))
all_nmse_db_max = np.zeros((SNRdB_values.shape))

it = 0
for SNRdB in SNRdB_values:
    # load datasets
    #print('Loading data...')
    #mat = loadmat(f"E:\\Wesin\\Doutorado\\channel-estimation\\datasets\\{quantization}\\testing\\mimo8x8_{SNRdB}snr_256samples_{frequency}.mat")
    #inputs, outputs = [mat[key] for key in ["inputs", "outputs"]]
    training_generator.SNRdB = SNRdB
    # get rid of the last example in the training_generator's memory (flush it)
    inputs, outputs = training_generator.get_examples(1)
    # now get the actual examples:
    inputs, outputs = training_generator.get_examples(num_test_examples)
    #savemat(f"C:\\Users\\wesin\\Documents\\Wesin\\Doutorado\\channel-estimation\\datasets\\1-bit\\testing\\generator\\mimo8x8_hq_{SNRdB}snr_3Ghz_4K.mat", {'inputs': inputs, 'outputs': outputs})
    #print("numExamples: ",len(outputs), np.mean(outputs), np.min(outputs), np.max(outputs))

    '''
    # normalizing values
    inputs[inputs < 0] = 0
    
    outputs = outputs.reshape((-1, 2 * Nt * Nr))

    scaler = MinMaxScaler()
    scaler.fit(outputs)

    outputs = scaler.transform(outputs)
    outputs = outputs.reshape((-1,2 * Nr, Nt))
    '''
    
    predictedOutput = model.predict(inputs)
    #tmp = outputs.reshape((-1, 2 * Nr * Nt))
    #outputs = scaler.fit_transform(tmp).reshape((-1,2 * Nr, Nt))
    error = outputs - predictedOutput
    #print(np.mean(predictedOutput), np.min(predictedOutput), np.max(predictedOutput))
    
    mseTest = np.mean(error[:] ** 2)
    print("overall MSE = ", mseTest)
    mean_nmse = mseTest / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)
    nmses = np.zeros((num_test_examples,))
    for i in range(num_test_examples):
        this_H = inputs[i]
        this_error = error[i]
        nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

    print("NMSE: mean", np.mean(nmses), "min", np.min(nmses), "max", np.max(nmses))
    nmses_db = 10 * np.log10(nmses)
    print(
        "NMSE dB: mean",
        np.mean(nmses_db),
        "min",
        np.min(nmses_db),
        "max",
        np.max(nmses_db),
    )

    all_nmse_db_average[it] = np.mean(nmses_db)
    all_nmse_db_min[it] = np.min(nmses_db)
    all_nmse_db_max[it] = np.max(nmses_db)

    it += 1

    del inputs
    del outputs
    #del mat 

output_filename = (
    f"all_nmse_mismatched_Nr{Nr}_Nt{Nt}_numEx256_rosslyn.txt"
)
output_filename = os.path.join(logdir, output_filename)
np.savetxt(output_filename, (all_nmse_db_average, all_nmse_db_min, all_nmse_db_max))
print("Wrote file", output_filename)
print("*******************\n{}".format(np.mean(all_nmse_db_average)))
