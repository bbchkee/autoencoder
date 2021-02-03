#!/usr/bin/python

# Use script by these arguments: input file, output file, model
# e.g:
# denoiser.py --noise signals_with_noise.npy --result denoised_signals.npy --model model_1.h5 --map mapping_info.npy
# input and output file format: NPY (numpy standart bit format)
import sys
import numpy as np
import argparse


# Define some useful functions
def processing_data(array):
    shifted_signals = []
    for signal in array:
        xmin = np.abs(np.min(list(signal)))
        signal = np.asarray(signal) + xmin
        shifted_signals.append(signal)
    maxs = np.max(shifted_signals)
    shifted_signals = shifted_signals / maxs
    return shifted_signals, maxs


# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('--no', '--noise', action='store', dest='noise')
parser.add_argument('-res', '--result', action='store', dest='result')
parser.add_argument('-mo', '--model', action='store', dest='model')
parser.add_argument('-ma', '--map', action='store', dest='mapping_info')
parser.add_argument('-te', '--test', action='store', dest='x_test')
parser.add_argument('-tn', '--tnoisy', action='store', dest='x_noisy')
args = parser.parse_args()

noised_signals = args.noise
model = args.model
result = args.result

# Process data
noised_signals = np.load(noised_signals)
print(noised_signals.shape)
# Predict pipeline
from keras.models import load_model
autoencoder = load_model(model)
shape_0 = noised_signals.shape[0]
shape_1 = noised_signals.shape[1]
predict = autoencoder.predict(np.reshape(noised_signals, (shape_0, shape_1, 1)))
predict = np.asarray(predict)

# Save results 
np.save('{}/x_denoised.npy'.format(result), predict)

