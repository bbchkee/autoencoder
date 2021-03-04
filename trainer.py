#!/usr/bin/python

# Used for create and train CCN via uDocker container
# Arguments: file without noise, file with noise, min, max, epochs
# e.g:
# trainer.py --signal signal.npy --noise noised_signal.npy --min 100 --max 200 --arch model.json --epochs 100
import sys

sys.path.append('home/python/denoiser')
import os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import argparse
from scipy.signal import hilbert

import keras
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization
from keras.models import Model

from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import Conv2D, Conv2DTranspose


def ae_relu1():
    input_img = Input(shape=(4096, 1))

    x = Conv1D(64, 32, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(128, 16, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(16, 128, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 256, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    relu1 = lambda x: keras.activations.relu(x, alpha=0.0, max_value=1.0, threshold=0)
    decoded = Conv1D(1, 8, activation='sigmoid', padding='same')(x)
    
    decoder = Model(input_img, decoded, name="decoder")
    autoencoder = Model(input_img, decoded, name="autoencoder")
    return decoder, autoencoder


def ae_nomaxpool():
    input_img = Input(shape=(4096, 1))
    input_img1 = keras.layers.Reshape((4096, 1, 1))(input_img)

    y = Conv2D(32, (32, 1), activation='relu', padding='same', strides=(2, 1))(input_img1)
    x = Conv2D(64, (16, 1), activation='relu', padding='same', strides=(2, 1))(y)
    
    x = Conv2DTranspose(16, (32, 1), activation='relu', padding='same', strides=(2, 1))(x)
    # x = keras.layers.Concatenate()([x, y])
    x = Conv2DTranspose(8, (64, 1), activation='relu', padding='same', strides=(2, 1))(x)
    relu1 = lambda x: keras.activations.relu(x, alpha=0.0, max_value=1.0, threshold=0)
    decoded = Conv2D(1, (8, 1), activation='linear', padding='same')(x)
    decoded = keras.layers.Reshape((4096, 1))(decoded)
    
    decoder = Model(input_img, decoded, name="decoder")
    autoencoder = Model(input_img, decoded, name="autoencoder")
    return decoder, autoencoder


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 40
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


def preprocessing(signal, noised_signal, min, max):
    signals_list = np.load(signal)
    noise_signals = np.load(noised_signal)
    # Only signals in min-max window
    signals = []
    noise = []
    for i, j in zip(signals_list, noise_signals):
        amp = np.max(abs(hilbert(i)))
        if amp < max and amp > min:
            signals.append(i)
            noise.append(j)
    signals = np.asarray(signals)
    noise = np.asarray(noise)

    # Normalization
    signals_shift = []
    noise_shift = []
    mapping = []
    amp_j_store = []

    for i, j in zip(signals, noise):
        i_m = 2 * np.max(abs(hilbert(i)))
        j_m = 2 * np.max(abs(hilbert(j)))

        # i_max, i_min = np.max(i), np.min(i)
        # j_max, j_min = np.max(j), np.min(j)
        # i = ((np.array(i)- j_min)/(j_max - j_min))

        i = (np.array(i)/j_m) + 0.5
        j = (np.array(j)/j_m) + 0.5
  
        signals_shift.append(i)
        noise_shift.append(j)
        mapping.append((j_m, j_m))
    #for i, j in zip(signals, noise):
    #    noise_j = np.std(j[noise_window])
    #    amp_i = np.max(abs(hilbert(i))[window])
    #    amp_j = np.max(abs(hilbert(j))[window])
    #    snr = amp_j/noise_j
    #    if snr > 3:
    #        continue
    #    else:
    #        i = (np.array(i)/(16*noise_j))+0.5
    #        j = (np.array(j)/(16*noise_j))+0.5
    #        amp_j_store.append(amp_j)
    #        if (np.min(j) or np.min(i)) < 0:
    #             print('Noise, amp noise, amp true, snr, max, min', noise_j, amp_j, amp_i, snr, np.max(j), np.min(j))
    #             break 
    #        signals_shift.append(i)
    #        noise_shift.append(j)
    #        mapping.append(noise_j)
   # print('Max amp', sorted(amp_j_store)[-1])
    maxs = np.max(signals_shift)
    maxn = np.max(noise_shift)
    print('Size of dataset', np.asarray(signals_shift).shape)
    return np.asarray(signals_shift), np.asarray(noise_shift), mapping, maxs, maxn


def create_model(signal, noised_signal, min, max):
    signals_shift, noise_shift, mapping, maxs, maxn = preprocessing(signal, noised_signal, min, max)
    print('maxs and maxn: ', maxs, maxn)
    x_train, x_test, x_train_noisy, x_test_noisy, mapping_train, mapping_test = train_test_split(signals_shift,
                                                                                                 noise_shift,
                                                                                                 mapping,
                                                                                                 random_state=42,
                                                                                                 test_size=0.2)
    # Write test arrays into files
    np.save('{}/x_test.npy'.format(directory), x_test)
    np.save('{}/x_test_noisy.npy'.format(directory), x_test_noisy)
    np.save('{}/mapping_info.npy'.format(directory), mapping_test)
    shape_0 = x_train.shape[0]
    shape_1 = x_train.shape[1]

    # with open(arch, 'r') as f:
    #     autoencoder = model_from_json(f.read())

    c_decoder, autoencoder = ae_nomaxpool()
    # 'binary_crossentropy'
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=opt, loss='mape', metrics=['mean_squared_error', 'mean_absolute_error'])

    autoencoder.fit(np.reshape(x_train_noisy, (shape_0, shape_1, 1)), np.reshape(x_train, (shape_0, shape_1, 1)),
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]
                    )

    autoencoder.save('{}/model.h5'.format(directory))
    return True


parser = argparse.ArgumentParser()
parser.add_argument('--wo', '--signal', action='store', dest='signal')
parser.add_argument('-wn', '--noise', action='store', dest='noise')
parser.add_argument('-mn', '--min', action='store', dest='min')
parser.add_argument('-mx', '--max', action='store', dest='max')
parser.add_argument('-ar', '--arch', action='store', dest='arch')
parser.add_argument('-ep', '--epochs', action='store', dest='epochs')
parser.add_argument('-sd', '--save_dir', action='store', dest='directory')
parser.add_argument('-en', '--exp_name', action='store', dest='exp_name')



args = parser.parse_args()

input_signals = str(args.signal)
input_noised_signals = str(args.noise)
min = int(args.min)
max = int(args.max)
arch = str(args.arch)

if not bool(args.epochs):
    epochs = 100
else:
    epochs = int(args.epochs)

ns = 0.325
time = np.arange(4096) * ns
noise_window = np.where((time < 550)| (time > 800))
window = np.where((time > 550)& (time < 800))


directory = args.directory + '/' + args.exp_name +'_' + arch.split('/')[-1] #'/home/soft/denoiser/datasets/' + '_' + arch.split('/')[-1]
print(directory)
if not os.path.isdir(directory):
  print("created directory")
  os.makedirs(directory)

create_model(input_signals, input_noised_signals, min, max)
