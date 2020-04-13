'''
#######################################################################
Script to train several models for channel estimation with DTL
in order to check sample complexity
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
    Conv2D,
    Conv1D,
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
import time
import psutil
import gc

p = psutil.Process(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "trte"], default="trte")
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--weights", type=str, default="akmodel")
args = parser.parse_args()

frequency = "60GHz"
quantization = "5-bit"

logdir = f"./mimo8x8/{quantization}/sample_complexity"
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
epochs = 10

# Parameters
global Nt
global Nr
Nt = 8  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
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
batch_size = 5  # numExamplesWithFixedChannel

#num_test_examples = [50, 60, 70, 80]  # for evaluating in the end, after training
#num_test_examples = [50, 60, 70, 80, 100, 120, 140, 160, 180, 200]  # for evaluating in the end, after training
#num_test_examples = np.linspace(20, 200, 20, dtype='int')  # for evaluating in the end, after training
num_test_examples = 200  # for evaluating in the end, after training

# get small number to avoid slowing down the simulation, test in the end
#num_validation_examples = [100, 110, 120, 130]
#num_validation_examples = [100, 110, 120, 130, 200, 220, 240, 260, 280, 300]
num_validation_examples = np.linspace(3, 30, 20, dtype='int')

#num_training_examples = [200, 250, 300, 350]
#num_training_examples = [200, 250, 300, 350, 400, 500, 600, 700, 800, 900]
num_training_examples = np.linspace(10, 100, 20, dtype='int')
file = "channel_china60Ghz.mat"
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

# real / compl as twice number of rows
input_shape = (numSamplesPerExample, 2 * (Nr))
output_dim = (2 * Nr, Nt)

numInputs = np.prod(input_shape)
numOutputs = np.prod(output_dim)
print(numInputs, " ", numOutputs)

cpu_time = {}
memory_consumption = {}

def train_model(input_train, output_train, input_val, output_val, instance, iteration):
    """train model"""

    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)

    matched_model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    matched_model.add(Conv1D(64,3,activation="tanh", padding = "same", input_shape=input_shape ))
    matched_model.add(Conv1D(64,3,activation="tanh", padding = "same"))
    matched_model.add(Dropout(0.3))
 
 
    matched_model.add(Flatten())
    #matched_model.add(Dense(150, activation="tanh"))
    matched_model.add(Dense(numOutputs, activation="linear"))
    matched_model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    matched_model.add(Reshape(output_dim))

    matched_model.compile(loss="mse", optimizer="adam")

    start_time = time.time()
    start_memory = p.memory_info().rss

    matched_model.fit(
        input_train,
        output_train,
        batch_size = 64,
        shuffle = True,
        validation_data = (input_val, output_val),
        epochs = epochs,
        verbose=1,    
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-7,
                patience=5,
                # restore_best_weights=True,
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

    cpu_time[str(instance) +"-"+ str(iteration)] = (time.time() - start_time)
    memory_consumption[str(instance) +"-"+ str(iteration)] = (p.memory_info().rss - start_memory)
    
    matched_model.save_weights(f'models/{quantization}/sample_complexity/matched_model_weights{instance}_iter{iteration}.h5')

    return matched_model

def test_model(my_model, num_test_example):
    ''' compute snrdb values and save in a file'''
    # test with disjoint test set
    SNRDB = 1
    training_generator.randomize_SNR = False
    
    # evaluate for several SNR values
    training_generator.SNRdB = SNRDB
    # get rid of the last example in the training_generator's memory (flush it)
    input_test, output_test = training_generator.get_examples(1)
    # get bottleneck features
    input_test, output_test = training_generator.get_examples(num_test_example)
    #input_test = bottleneck_model.predict(test_input)
    
    # now get the actual examples:
    #predicted_output = model.predict(input_test)
    predicted_output = my_model.predict(input_test)
    error = output_test - predicted_output
    
    mse_test = np.mean(error[:] ** 2)
    print("overall MSE = ", mse_test)
    mean_nmse = mse_test / (Nr * Nt)
    print("overall NMSE = ", mean_nmse)
    nmses = np.zeros((num_test_example,))
    for i in range(num_test_example):
        this_H = input_test[i]
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
    
    all_nmse_db_average = np.mean(nmses_db)
    all_nmse_db_min = np.min(nmses_db)
    all_nmse_db_max = np.max(nmses_db)

    return all_nmse_db_average, all_nmse_db_min, all_nmse_db_max

for number in range(len(num_training_examples)):

    all_nmse_avg, all_nmse_min, all_nmse_max = [], [], []
    nmse_avg, nmse_min, nmse_max = 0, 0, 0

    for it in range(5):

        training_generator.randomize_SNR = True
        training_generator.min_randomized_snr_db = min_randomized_snr_db
        training_generator.max_randomized_snr_db = max_randomized_snr_db

        y_train, h_train = training_generator.get_examples(num_training_examples[number])
        y_val, h_val = training_generator.get_examples(num_validation_examples[number])

        new_model = train_model(y_train, h_train, y_val, h_val, num_training_examples[number], it)
        test_model(new_model, num_test_examples, )
        nmse_avg, nmse_min, nmse_max = test_model(new_model, num_test_examples)
        all_nmse_avg.append(nmse_avg)
        all_nmse_min.append(nmse_min)
        all_nmse_max.append(nmse_max)

        del h_train, y_train, h_val, y_val
        del new_model

        gc.collect()
        K.reset_uids()
        K.clear_session()

    # write file
    # output_filename = 'all_nmse_snrdb_' + str(SNRdB) + '_Nr' + str(Nr) + '_Nt' + str(Nt) + '_numEx' + str(
    #    numSamplesPerExample) + '.txt'
    output_filename = (
        f"mean_nmse_matched_Nr{Nr}_Nt{Nt}_numEx{numSamplesPerExample}_beijing_{number}.txt"
    )
    output_filename = os.path.join(logdir, output_filename)
    np.savetxt(output_filename, (np.mean(all_nmse_avg), np.mean(all_nmse_min), np.mean(all_nmse_max)))

    print("Wrote file", output_filename)

with open(os.path.join(logdir, "complexity_matched_10-100.txt"), 'w') as f:
    f.write("Time complecity on traning\n")
    f.write("{}".format(cpu_time))
    f.write("\nMemory complexity on training\n")
    f.write("{}".format(memory_consumption))
