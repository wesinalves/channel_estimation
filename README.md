# Paper

Tensorflow implementation of [Deep Transfer Learning for Site-Specific Channel
Estimation in Low-Resolution mmWave MIMO](https://ieeexplore.ieee.org/document/9388873)

*Abstract* — We consider the problem of channel estimation in
low-resolution multiple-input multiple-output (MIMO) systems
operating at millimeter wave (mmWave) and present a deep
transfer learning (DTL) approach that exploits previously trained
models to speed up site adaptation. The proposed model is
composed of a feature extractor and a regressor, with only the
regressor requiring training for the new environment. The DTL
approach is evaluated using two 3D scenarios where ray-tracing
is performed to generate the mmWave MIMO channels used in
the simulations. Under the defined testing setup, the proposed
DTL approach can reduce the computational cost of the training
stage without decreasing the estimation accuracy.

![training_sample](img/sample.png)

# Usage

First, run ```python wr_fres.py``` to train models.

Then, run ```python pretrained_model.py``` to apply deep transfer learning

# Directory description

*akpy* - software to process mimo channels

*mimoNrxNt* - experiments outputs

*models* - save models trained

Filename   | Description
--------- | ------
wr_fres.py | main file to train models
pretrained_model.py | apply deep transfer learning in trained models
get_channels.py | extract channels data from Raymobtime dataset
mimo_channels.py | base file to generate mimo channels
mimo_channels_data_generator2.py | this file generates input data for the model training
mimo8x32/plot_* | plot graphs shown in paper


# Datasets

*channel_rosslyn60Ghz* - 8x8 channels in Rosslyn.

*channels_rosslyn_60Ghz_Nr8Nt32_mobile_s004* - 8x32 mobile channels in Rosslyn.

*channels_beijing_60Ghz_Nr8Nt32_mobile_s007* - 8x32 mobile channels in Beijing.

See more details in [Raymobtime dataset](https://www.lasse.ufpa.br/raymobtime/).