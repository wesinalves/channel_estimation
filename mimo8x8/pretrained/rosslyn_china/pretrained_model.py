'''
#######################################################################
Script to create pre-trained models for channel estimation with DL
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

p = psutil.Process(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "trte"], default="trte")
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--weights", type=str, default="akmodel")
args = parser.parse_args()

frequency = "60GHz"
quantization = "1-bit"

logdir = "./mimo8x8/pretrained"
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
num_test_examples = [100, 120, 140, 160, 180, 200]  # for evaluating in the end, after training
# get small number to avoid slowing down the simulation, test in the end
#num_validation_examples = [100, 110, 120, 130]
num_validation_examples = [200, 220, 240, 260, 280, 300]
#num_training_examples = [200, 250, 300, 350]
num_training_examples = [400, 500, 600, 700, 800, 900]
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

if True:
    training_generator.randomize_SNR = True
    training_generator.min_randomized_snr_db = min_randomized_snr_db
    training_generator.max_randomized_snr_db = max_randomized_snr_db
else:
    training_generator.randomize_SNR = True
    training_generator.SNRdB = 0

model_filepath = "models/model-rosslyn-conv.h5"
model = keras.models.load_model(model_filepath)
SNRDB_VALUES = np.arange(-21, 22, 3)


all_nmse_db_average = np.zeros((SNRDB_VALUES.shape))
all_nmse_db_min = np.zeros((SNRDB_VALUES.shape))
all_nmse_db_max = np.zeros((SNRDB_VALUES.shape))


#model.compile(loss="mse", optimizer="adam")
# model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
print("loaded model from file")
print(model.summary())

# real / compl as twice number of rows
input_shape = (numSamplesPerExample, 2 * (Nr))
output_dim = (2 * Nr, Nt)

numInputs = np.prod(input_shape)
numOutputs = np.prod(output_dim)
print(numInputs, " ", numOutputs)

cpu_time = {}
memory_consumption = {}

def save_bottleneck_features(input_train, input_val, baseline, instance):
    '''Save features extracted from internal layers of the model'''
    # remove top layers from baseline model
    bottleneck_model = Sequential()
    for i,layer in enumerate(baseline.layers[:-4]):
        print(layer.name, i)
        bottleneck_model.add(layer)
        bottleneck_model.layers[i].set_weights(layer.get_weights())
    
    bottleneck_model.summary()

    predicted_output = bottleneck_model.predict(input_train)
    # save features
    with open('models/pretrained/bottleneck_features_train{}.npy'.format(instance), 'wb') as file1:
        np.save(file1, predicted_output)

    predicted_val = bottleneck_model.predict(input_val)
    with open('models/pretrained/bottleneck_features_val{}.npy'.format(instance), 'wb') as file2:
        np.save(file2, predicted_val)

    return bottleneck_model
    


def train_top_model(output_train, output_val, instance):
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)

    
    # load bottleneck features
    with open('models/pretrained/bottleneck_features_train{}.npy'.format(instance), 'rb') as file1:
        input_train = np.load(file1)
    with open('models/pretrained/bottleneck_features_val{}.npy'.format(instance), 'rb') as file2:
        input_val = np.load(file2)

    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(numOutputs, activation="linear"))
    top_model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    top_model.add(Reshape(output_dim))

    top_model.compile(loss="mse", optimizer="adam")

    start_time = time.time()
    start_memory = p.memory_info().rss

    top_model.fit(input_train, output_train,
                epochs = epochs,
                batch_size = 64,
                #shuffle = True,
                validation_data = (input_val, output_val),
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
    
    cpu_time[str(instance)] = (time.time() - start_time)
    memory_consumption[str(instance)] = (p.memory_info().rss - start_memory)
    
    top_model.save_weights('models/pretrained/bottleneck_fc_model_weights{}.h5'.format(instance))

    return top_model
  
def test_model(bottleneck_model, top_model, num_test_example):
    ''' compute snrdb values and save in a file'''
    # test with disjoint test set
    SNRDB_VALUES = np.arange(-21, 22, 3)
    training_generator.randomize_SNR = False
    new_model = Sequential()
    new_model.add(bottleneck_model)
    new_model.add(top_model)

    # evaluate for several SNR values
    it = 0
    for SNRDB in SNRDB_VALUES:
        training_generator.SNRdB = SNRDB
        # get rid of the last example in the training_generator's memory (flush it)
        input_test, output_test = training_generator.get_examples(1)
        # get bottleneck features
        input_test, output_test = training_generator.get_examples(num_test_example)
        #input_test = bottleneck_model.predict(test_input)
        
        # now get the actual examples:
        #predicted_output = model.predict(input_test)
        predicted_output = new_model.predict(input_test)
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
        if True:
            all_nmse_db_average[it] = np.mean(nmses_db)
        else:
            all_nmse_db_average[it] = mean_nmse  # np.mean(nmses_db)
        all_nmse_db_min[it] = np.min(nmses_db)
        all_nmse_db_max[it] = np.max(nmses_db)

        # write file
    
        it += 1

    # write file
    # output_filename = 'all_nmse_snrdb_' + str(SNRdB) + '_Nr' + str(Nr) + '_Nt' + str(Nt) + '_numEx' + str(
    #    numSamplesPerExample) + '.txt'
    output_filename = (
        f"all_nmse_pretrained_Nr{Nr}_Nt{Nt}_numEx{numSamplesPerExample}_rosslyn_china_{num_test_example}.txt"
    )
    output_filename = os.path.join(logdir, output_filename)
    np.savetxt(output_filename, (all_nmse_db_average, all_nmse_db_min, all_nmse_db_max))

    print("Wrote file", output_filename)
    print("*******************\n{}".format(np.mean(all_nmse_db_average)))


for i in range(len(num_training_examples)):

    x_train, y_train = training_generator.get_examples(num_training_examples[i])
    x_val, y_val = training_generator.get_examples(num_validation_examples[i])

    bottleneck_model = save_bottleneck_features(x_train, x_val, model, num_training_examples[i])
    top_model = train_top_model(y_train, y_val, num_training_examples[i])
    test_model(bottleneck_model, top_model, num_test_examples[i])

with open(os.path.join(logdir, "complexity100-200.txt"), 'w') as f:
    f.write("Time complecity on traning\n")
    f.write("{}".format(cpu_time))
    f.write("\nMemory complexity on training\n")
    f.write("{}".format(memory_consumption))

