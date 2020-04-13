"""
Performs channel estimation using neural networks.
This version:

    * makes a loop over SNR values
    OBS: Tensorflow is not working probably because it's being repeatedly initialized in the loop
"""
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
    Conv1D,
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
import time
#import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "trte"], default="trte")
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--weights", type=str, default="akmodel")
args = parser.parse_args()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_channel_unstructured3"
logdir = "{}/run-{}/".format(root_logdir, now)

logdir = "./mimo8x8/1-bit"
# save this script
os.makedirs(logdir, exist_ok=True)
ini = sys.argv[0]
shutil.copyfile(sys.argv[0], os.path.join(logdir, os.path.basename(sys.argv[0])))
print("Copied:", sys.argv[0], "to", os.path.join(logdir, os.path.basename(sys.argv[0])))

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


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

num_test_examples = 100  # for evaluating in the end, after training
# get small number to avoid slowing down the simulation, test in the end
num_validation_examples = 200
num_training_examples = 900
file = "channel_rosslyn60Ghz.mat"
if False:
    num_clusters = 1
    method = "sparse"
else:
    method = "manual"
    #method = "randomized_sparse"
    # method = 'random' #not working, Hv is not calculated I guess
# Generator
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

input_train, output_train = training_generator.get_examples(num_training_examples)
input_val, output_val = training_generator.get_examples(num_validation_examples)

global H_normalization_factor
H_normalization_factor = np.sqrt(Nr * Nt)

global K_H_normalization_factor
K_H_normalization_factor = K.variable(H_normalization_factor, dtype=np.float32)

if False:
    # real / compl as new dimension
    input_shape = (Nr, numSamplesPerExample, 2)
    output_dim = (Nr, Nt, 2)
else:
    # real / compl as twice number of rows
    input_shape = (numSamplesPerExample, 2 * (Nr))
    output_dim = (2 * Nr, Nt)

numInputs = np.prod(input_shape)
numOutputs = np.prod(output_dim)
print(numInputs, " ", numOutputs)

def create_conv2D():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    model = Sequential()
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Reshape((2 * Nr, numSamplesPerExample, 1), input_shape=input_shape))
    model.add(
        Conv2D(
            80,
            kernel_size=(2, numSamplesPerExample),
            activation="tanh",
            # 				 strides=[1,1],
            # 				 padding="SAME",
        )
    )
    model.add(Conv2D(30, (numSamplesPerExample, 2), padding="SAME", activation="tanh"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(20, (2, 2), padding="SAME", activation='tanh'))
    # model.add(Dropout(0.3))
    model.add(Dense(30, activation="tanh"))
    # model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


def create_dense():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # K_H_normalization_factor = K.variable(H_normalization_factor,dtype=np.float32)

    N = 150
    model = Sequential()
    model.add(Dense(N, activation="tanh", input_shape=input_shape))
    model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(Dense(N, activation="tanh"))
    # model.add(BatchNormalization())
    # model.add(Dense(100, activation='tanh'))
    # model.add(BatchNormalization())
    if False:
        model.add(Dense(100, activation="tanh"))
        model.add(BatchNormalization())
        model.add(Dense(100, activation="tanh"))
        model.add(BatchNormalization())
        model.add(Dense(100, activation="tanh"))
        model.add(BatchNormalization())
        model.add(Dense(100, activation="tanh"))
        model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(numOutputs, activation='linear',kernel_constraint=max_norm(1.)))
    model.add(Dense(numOutputs, activation="linear"))
    # model.add(BatchNormalization(center=False))
    # https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize
    # model.add(Lambda(lambda x: K_H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    # model.add(Lambda(lambda x: x / K.sqrt(K.sum(x**2))))
    # model.add(Lambda(normalize_output), output_shape=(numOutputs,))
    model.add(Reshape(output_dim))
    return model


def create_conv():
    model = Sequential()
    model.add(Reshape((2 * Nr, numSamplesPerExample, 1), input_shape=input_shape))
    model.add(
        Conv2D(
            100,
            kernel_size=(1, numSamplesPerExample),
            activation="tanh",
            # 				 strides=[1,1],
            # 				 padding="SAME",
        )
    )  # input_shape=input_shape
    model.add(Conv2D(10, (numSamplesPerExample, 1), padding="SAME", activation="tanh"))
    # model.add(MaxPooling2D(pool_size=(6, 6)))
    model.add(Conv2D(5, (4, 4), padding="SAME", activation="tanh"))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(80, activation="tanh"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(60, activation="tanh"))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    model.add(Dense(80, activation="tanh"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Dense(80, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(3, activation='softmax'))
    model.add(Dense(numOutputs, activation="linear"))
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


def create_res():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    # residual net using Kera's functional API (not using Sequential)
    N = 2
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


def create_lstm():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    model = Sequential()

    model.add(lstm(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(lstm(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(lstm(256, return_sequences=False))
    # model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


def create_lstm_attention():
    global Nt
    global Nr
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    inputs = Input(input_shape)

    activations = lstm(32, return_sequences=True)(inputs)
    activations = BatchNormalization()(activations)
    # for i in range(1):
    # activations = lstm(64, return_sequences=True)(activations)

    # compute importance for each step
    attention = keras.layers.Permute([2, 1])(activations)
    attention = Dense(numSamplesPerExample, activation="softmax")(attention)
    if False:
        attention = Lambda(lambda x: K.mean(x, axis=1))(attention)
        attention = keras.layers.RepeatVector(numSamplesPerExample)(attention)
    a_probs = keras.layers.Permute((2, 1))(attention)

    sent_representation = keras.layers.merge.multiply([activations, a_probs])
    sent_representation = Flatten()(sent_representation)
    d = sent_representation
    d = BatchNormalization()(d)
    # for neurons in [100,100,100,100]:
    #    d = Dense(neurons, activation='relu')(d)
    y3 = Dense(numOutputs, activation="linear")(d)
    y4 = Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1))(y3)
    y5 = Reshape(output_dim)(y4)

    model = Model(inputs=inputs, outputs=y5)
    return model


def create_lstm_conv():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)

    conv1D = keras.layers.Conv1D
    lstm = keras.layers.CuDNNLSTM
    # lstm = keras.layers.LSTM

    timeDistributed = keras.layers.TimeDistributed
    use_batch_norm = True
    use_activation = True
    use_dropout = True

    dropout_rate = 0.3
    activation = "relu"

    conv1DConfig = [
        # (96, 8, 1),
        (48, 5, 1),
        (64, 5, 1),
        # (96, 6, 1),
        # (80, 4, 2),
        (48, 3, 2),
    ]
    conv1DConfig = [(4 * x, y, z) for x, y, z in conv1DConfig]
    lstmConfig = []

    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    first = True
    for filt, kern, strides in conv1DConfig:
        model.add(conv1D(filt, kern, strides=strides, activation="relu"))
        if use_batch_norm:
            model.add(BatchNormalization())
        # if use_activation:
        #    model.add(keras.layers.Activation(activation))
        if use_dropout:
            if first:
                first = False
            else:
                model.add(Dropout(dropout_rate))

    for units in lstmConfig:
        model.add(keras.layers.Bidirectional(lstm(units, return_sequences=True)))

    # model.add(timeDistributed(Dense(1, activation='relu')))
    # model.add(keras.layers.Permute((2, 1)))
    # model.add(timeDistributed(Dense(8, activation=activation)))
    model.add(Flatten())
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model


def create_deep_dense():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)

    num_hidden_layers = 8
    activation = "relu"

    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(Flatten())

    use_dropout = False
    use_batch_norm = False
    num_neurons = 128

    for i in range(num_hidden_layers):
        model.add(Dense(num_neurons, activation=activation))
        if use_batch_norm:
            model.add(BatchNormalization())
        if use_dropout and i != 0 and i != (num_hidden_layers - 1):
            model.add(Dropout(0.3))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
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

def create_conv3():
    global Nr
    global Nt
    H_normalization_factor = np.sqrt(Nr * Nt)
    lstm = keras.layers.CuDNNLSTM

    model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Conv1D(64,3,activation="tanh", padding = "same", input_shape=input_shape ))
    model.add(Conv1D(64,3,activation="tanh", padding = "same"))
    #model.add(AveragePooling1D(2))
    model.add(Dropout(0.3))

   
    model.add(Flatten())
    #model.add(Dense(150, activation="tanh"))
    model.add(Dense(numOutputs, activation="linear"))
    model.add(Lambda(lambda x: H_normalization_factor * K.l2_normalize(x, axis=-1)))
    model.add(Reshape(output_dim))
    return model



# define baseline model, avoid relu activation given that we have negative numbers
def baseline_model():
    # return create_dense()
    # return create_conv()
    return create_conv2()
    # return create_res()
    # return create_res2()
    # return create_res3()
    # return create_lstm()
    # return create_lstm_attention()
    # return create_deep_dense()
    #return create_lstm_conv()

weights_filename = "model_{}_weights_channel_estimation_rosslyn".format(1)
model_filepath = "models/1-bit/model-rosslyn-conv.h5"
model = baseline_model()
# Compile model
# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'    'cosine'])
model.compile(loss="mse", optimizer="adam")
# model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=0.0001,  clipvalue=1.0,decay=1e-6))
if args.mode == "train":
    model = keras.models.load_model(model_filepath)
    model.compile(loss="mse", optimizer="adam")
    # model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
    print("loaded model from file")
print(model.summary())

for layer in model.layers:
    print(layer.name)

embedding_layer_names = set(
    layer.name for layer in model.layers if layer.name.startswith("dense_")
)

start_time = time.time()
#total_memory = 0

# Train model on dataset
if args.mode == "train" or args.mode == "trte":
    if False:
        tensorboard = TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            # batch_size=32,
            write_graph=True,
            write_grads=True,
            write_images=True,
            embeddings_freq=10,
            embeddings_metadata=None,
            embeddings_layer_names=embedding_layer_names,
        )

        history = model.fit_generator(
            generator=training_generator,
            validation_data=(input_val, output_val),
            epochs=epochs,
            verbose=1,
            use_multiprocessing=False,
            callbacks=[tensorboard],
        )
    else:
        history = model.fit(input_train, output_train,
            validation_data=(input_val, output_val),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            use_multiprocessing=False,
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

    print("Num of used channels over train: ", training_generator.num_channel_used)
    # serialize model to JSON
    # model_json = model.to_json()
    # with open(os.path.join(logdir, weights_filename + ".json"), "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights(os.path.join(logdir, weights_filename + ".h5f"))

    # history_dictionary = history.history
    # Save it under the form of a json file
    # json.dump(
    #    history_dictionary,
    #    open(os.path.join(logdir, weights_filename + ".history_dic"), "w"),
    # )
    # Can then read the dict with:
    # json_file = open('vq_1_weights_channel_estimation.history_dic', 'r')
    # history_dictionary = json_file.read()
if args.mode == "test" or args.mode == "trte":
    # load json and create model
    # json_file = open(os.path.join(logdir, weights_filename + ".json"), "r")
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(os.path.join(logdir, weights_filename + ".h5f"))
    model = keras.models.load_model(model_filepath)
    # test with disjoint test set
    # SNRdB_values = np.arange(-30,20)
    SNRdB_values = np.arange(-21, 22, 3)
    training_generator.randomize_SNR = False
    if True:
        training_generator.method = "manual"
        #training_generator.method = "randomized_sparse"
        # training_generator.method = 'random'
    else:
        training_generator.method = "sparse"
        training_generator.num_clusters = 3

    all_nmse_db_average = np.zeros((SNRdB_values.shape))
    all_nmse_db_min = np.zeros((SNRdB_values.shape))
    all_nmse_db_max = np.zeros((SNRdB_values.shape))

    # evaluate for several SNR values
    it = 0
    for SNRdB in SNRdB_values:
        training_generator.SNRdB = SNRdB
        # get rid of the last example in the training_generator's memory (flush it)
        testInput, testOutput = training_generator.get_examples(1)
        # now get the actual examples:
        testInput, testOutput = training_generator.get_examples(num_test_examples)
        predictedOutput = model.predict(testInput)
        error = testOutput - predictedOutput
        if False:
            for ii in range(10):
                print("Norm test = ", la.norm(testOutput[ii]))
                print("Norm predicted = ", la.norm(predictedOutput[ii]))

        if False:
            print("Norm test = ", la.norm(testOutput[0]))
            print("Norm predicted = ", la.norm(predictedOutput[0]))
            print("Norm error = ", la.norm(testOutput[0] - predictedOutput[0]))
            plt.subplot(1, 3, 1)
            plt.imshow(10 * np.log10(testOutput[0]))
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(10 * np.log10(predictedOutput[0]))
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(10 * np.log10(testOutput[0] - predictedOutput[0]))
            plt.colorbar()
            plt.show()

        mseTest = np.mean(error[:] ** 2)
        print("overall MSE = ", mseTest)
        mean_nmse = mseTest / (Nr * Nt)
        print("overall NMSE = ", mean_nmse)
        nmses = np.zeros((num_test_examples,))
        for i in range(num_test_examples):
            this_H = testInput[i]
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
        #total_memory += p.memory_info().rss

        # write file
        '''
        output_filename = (
            "nmse_snrdb_"
            + str(SNRdB)
            + "_Nr"
            + str(Nr)
            + "_Nt"
            + str(Nt)
            + "_numEx"
            + str(numSamplesPerExample)
            + ".txt"
        )
        f = open(output_filename, "w")
        f.write(str(nmses))
        f.close()
        print("Wrote file", output_filename)
        '''
        it += 1

    # write file
    # output_filename = 'all_nmse_snrdb_' + str(SNRdB) + '_Nr' + str(Nr) + '_Nt' + str(Nt) + '_numEx' + str(
    #    numSamplesPerExample) + '.txt'
    output_filename = (
        f"all_nmse_matched_Nr{Nr}_Nt{Nt}_numEx{numSamplesPerExample}_rosslyn.txt"
    )
    output_filename = os.path.join(logdir, output_filename)
    np.savetxt(output_filename, (all_nmse_db_average, all_nmse_db_min, all_nmse_db_max))
    # with open(output_filename, 'w') as f:
    #    f.write('Average=')
    #    f.write(str(all_nmse_db_average))
    #    f.write('Min=')
    #    f.write(str(all_nmse_db_min))
    #    f.write('Max=')
    #    f.write(str(all_nmse_db_max))
    print("Wrote file", output_filename)
    cpu_time =  (time.time() - start_time)
    print("*******************\n{}".format(np.mean(all_nmse_db_average)))
    #print("time ", cpu_time, "memory: ", total_memory)
    print("time ", cpu_time)

    # evaluate for several # of clusters with a fixed SNR
    if False:  # training_generator.method != 'random':
        # num_cluster_values = np.arange(1,Nt*Nr,4)
        num_cluster_values = np.arange(2, Nt * Nr, 3)
        all_clusters_average = np.zeros((num_cluster_values.shape))
        all_clusters_min = np.zeros((num_cluster_values.shape))
        all_clusters_max = np.zeros((num_cluster_values.shape))
        SNRdB = 0
        training_generator.SNRdB = SNRdB
        training_generator.method = "sparse"
        it = 0
        for this_num_cluster in num_cluster_values:
            training_generator.num_clusters = this_num_cluster
            testInput, testOutput = training_generator.get_examples(num_test_examples)
            predictedOutput = model.predict(testInput)
            error = testOutput - predictedOutput
            mseTest = np.mean(error[:] ** 2)
            print("overall MSE = ", mseTest)
            nmses = np.zeros((num_test_examples,))
            for i in range(num_test_examples):
                this_H = testInput[i]
                this_error = error[i]
                nmses[i] = np.mean(this_error[:] ** 2) / np.mean(this_H[:] ** 2)

            print(
                "NMSE: mean", np.mean(nmses), "min", np.min(nmses), "max", np.max(nmses)
            )
            nmses_db = 10 * np.log10(nmses)
            print(
                "NMSE dB: mean",
                np.mean(nmses_db),
                "min",
                np.min(nmses_db),
                "max",
                np.max(nmses_db),
            )
            all_clusters_average[it] = np.mean(nmses_db)
            all_clusters_min[it] = np.min(nmses_db)
            all_clusters_max[it] = np.max(nmses_db)

            # write file
            output_filename = (
                "clusters_nmse_numpaths_"
                + str(this_num_cluster)
                + "snrdb_"
                + str(SNRdB)
                + "_Nr"
                + str(Nr)
                + "_Nt"
                + str(Nt)
                + "_numEx"
                + str(numSamplesPerExample)
                + ".txt"
            )
            f = open(output_filename, "w")
            f.write(str(nmses))
            f.close()
            print("Wrote file", output_filename)
            it += 1

        # write file
        output_filename = (
            "clusters_all_nmse_snrdb_"
            + str(SNRdB)
            + "_Nr"
            + str(Nr)
            + "_Nt"
            + str(Nt)
            + "_numEx"
            + str(numSamplesPerExample)
            + ".txt"
        )
        f = open(output_filename, "w")
        f.write("Average=")
        f.write(str(all_clusters_average))
        f.write("Min=")
        f.write(str(all_clusters_min))
        f.write("Max=")
        f.write(str(all_clusters_max))
        f.close()

        print("Wrote file", output_filename)

