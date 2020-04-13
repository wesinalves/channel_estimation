from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np

from mimo_channels_data_generator2 import RandomChannelMimoDataGenerator

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

#num_samples = [10, 20, 50, 60, 100, 150, 200, 220, 240, 256]
num_samples = [256]
files = ["channel_data3GHz.mat", "channel_data5GHz.mat", "channel_data60GHz.mat", 
    "channel_data60GHz_mob.mat"]

#files = ["channel_data60GHz.mat"]

short_files = ["3GHz", "5GHz", "60GHz", "60GHz_mob"]
#short_files = ["60GHz"]

SNRs = np.arange(-21,22,3)

# common parameters
Nt = 8  # num of Rx antennas, will be larger than Nt for uplink massive MIMO
Nr = 8  # num of Tx antennas
min_randomized_snr_db = -1
max_randomized_snr_db = 1
batch_size = 5  # numExamplesWithFixedChannel
method = "manual"
num_training = 400
num_testing = 200
numExamplesWithFixedChannel = 1

index = 0

print("Extracting dataset...")
for file in files:
    
    for samples in num_samples:
        numSamplesPerExample = samples
        numSamplesPerFixedChannel = (
            numExamplesWithFixedChannel * numSamplesPerExample
        )  # 

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

        training_generator.randomize_SNR = True
        training_generator.min_randomized_snr_db = min_randomized_snr_db
        training_generator.max_randomized_snr_db = max_randomized_snr_db

        inputs, outputs = training_generator.get_examples(num_training)

        savemat(f"C:\\Users\\wesin\\Documents\\Wesin\\Doutorado\\channel-estimation\\datasets\\noquant\\mimo8x8_{samples}samples_{short_files[index]}.mat", {'inputs': inputs, 'outputs': outputs})
        

        for SNRdb in SNRs:
        
            training_generator.randomize_SNR = False
            training_generator.SNRdB = SNRdb
            #training_generator.min_randomized_snr_db = min_randomized_snr_db
            #training_generator.max_randomized_snr_db = max_randomized_snr_db

            inputs, outputs = training_generator.get_examples(num_testing)

            savemat(f"C:\\Users\\wesin\\Documents\\Wesin\\Doutorado\\channel-estimation\\datasets\\noquant\\testing\\mimo8x8_{SNRdb}snr_{samples}samples_{short_files[index]}.mat", {'inputs': inputs, 'outputs': outputs})

    index += 1

    
print("Extracting done...")



