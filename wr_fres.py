"""
Performs channel estimation using neural networks.
This version:

    * makes a loop over SNR values
    OBS: Tensorflow is not working probably because it's being repeatedly initialized in the loop
"""
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator

from tensorflow.keras.callbacks import TensorBoard

from matplotlib import pyplot as plt
import argparse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.constraints import max_norm
import numpy.linalg as la
from tensorflow.keras import backend as K
import os
import shutil
import sys
import json
from datetime import datetime
import time
#import psutil

from utils import baseline_model

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "trte"], default="trte")
# parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument("--weights", type=str, default="akmodel")
args = parser.parse_args()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_channel_unstructured3"
logdir = "{}/run-{}/".format(root_logdir, now)

logdir = "./mimo8x32/5-bit"
# save this script
os.makedirs(logdir, exist_ok=True)
ini = sys.argv[0]
shutil.copyfile(sys.argv[0], os.path.join(logdir, os.path.basename(sys.argv[0])))
print("Copied:", sys.argv[0], "to", os.path.join(logdir, os.path.basename(sys.argv[0])))

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# training parameters
epochs = 100

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
batch_size = 5  # numExamplesWithFixedChannel

num_test_examples = 100  # for evaluating in the end, after training
# get small number to avoid slowing down the simulation, test in the end
num_validation_examples = 200
num_training_examples = 900
file = "./channels_rosslyn_60Ghz_Nr8Nt32_mobile_s004.mat"
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

# global H_normalization_factor
# H_normalization_factor = np.sqrt(Nr * Nt)

# global K_H_normalization_factor
# K_H_normalization_factor = K.variable(H_normalization_factor, dtype=np.float32)

if False:
    # real / compl as new dimension
    input_shape = (Nr, numSamplesPerExample, 2)
    output_dim = (Nr, Nt, 2)
else:
    # real / compl as twice number of rows
    input_shape = (numSamplesPerExample, 2 * (Nr))
    output_dim = (2 * Nr, Nt)

# numInputs = np.prod(input_shape)
# numOutputs = np.prod(output_dim)
# print(numInputs, " ", numOutputs)

weights_filename = "model_{}_weights_channel_estimation_beijing".format(1)
model_filepath = "models/5-bit/model-beijing-conv.h5"
model = baseline_model(Nr,Nt, input_shape, output_dim)
model.compile(loss="mse", optimizer="adam")

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
        f"all_nmse_matched_Nr{Nr}_Nt{Nt}_numEx{numSamplesPerExample}_beijing.txt"
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

