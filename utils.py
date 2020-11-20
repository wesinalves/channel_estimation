import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
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

def create_conv2(Nr,Nt,input_shape, output_dim):
    H_normalization_factor = np.sqrt(Nr * Nt)
    # lstm = tf.keras.layers.CuDNNLSTM

    model = Sequential()
    
    # Keras is requiring an extra dimension: I will add it with a reshape layer because I am using a generator
    model.add(Conv1D(64,3,activation="tanh", padding = "same", input_shape=input_shape ))
    model.add(Conv1D(64,3,activation="tanh", padding = "same"))
    model.add(Dropout(0.3))
 
 
    model.add(Flatten())
    #model.add(Dense(150, activation="tanh"))
    model.add(Dense(np.prod(output_dim), activation="linear"))
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
def baseline_model(Nr,Nt,input_shape, output_dim):
    # return create_dense()
    # return create_conv()
    return create_conv2(Nr,Nt, input_shape, output_dim)
    # return create_res()
    # return create_res2()
    # return create_res3()
    # return create_lstm()
    # return create_lstm_attention()
    # return create_deep_dense()
    #return create_lstm_conv()

def normalize_output(x):
    print("x1=", x)
    norm = np.sum(x ** 2)
    # x = x / np.sqrt(norm)
    x = x / norm

    print("x2=", x)
    print(norm)
    return x