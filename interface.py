#!/usr/bin/python

# Use script by these arguments: input file, model file, output_file
# e.g:
# denoiser.py --noise signals_with_noise.npy --output denoised_signals.npy --model model_1.h5
import argparse
import sys
import numpy as np
from scipy.signal import hilbert


def read_file(txt):
    noised_data = []
    f = open(txt)
    for line in f:
        noised_data.append([float(x) for x in line.replace('\n', '').split(',')])
    f.close()
    return np.array(noised_data)


def get_peak_info(time, trace, window):
    i = np.argmax(np.abs(trace[window]))
    return max(abs(trace)[window]), time[i] + time[window][0]


# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('--no', '--noise', action='store', dest='noise')
parser.add_argument('-res', '--output', action='store', dest='output')
parser.add_argument('-mo', '--model', action='store', dest='model')
args = parser.parse_args()

noised_signals = args.noise
model = args.model
output = args.output

# Process data
noised_signals = read_file(noised_signals)
print(noised_signals.shape)
ns = 0.325
time = np.arange(len(noised_signals[0])) * ns
window = np.where((time > 550) & (time < 800))

noised_signals_shifted = []
mapping_info = []
for j in noised_signals:
    j_m = 2 * np.max(abs(hilbert(j)))
    j = (np.array(j) / j_m) + 0.5
    noised_signals_shifted.append(j)
    mapping_info.append(j_m)
noised_signals_shifted = np.array(noised_signals_shifted)



# Predict pipeline
from keras.models import load_model
autoencoder = load_model(model)
shape_0 = noised_signals_shifted.shape[0]
shape_1 = noised_signals_shifted.shape[1]
predict = autoencoder.predict(np.reshape(noised_signals_shifted, (shape_0, shape_1, 1)))
predict = np.asarray(predict)

peak_info = [get_peak_info(time, trace, window) for trace in predict]
peak_info = np.array(peak_info)

if peak_info[:,0] < 0.434:
    amplitudes = [0]
else:
    amplitudes = []
    for i, j in zip(peak_info[:, 0], mapping_info):
        amplitudes.append(((i-0.5)*j))

reco_peak_info = peak_info[:,1]
amplitudes = np.array(amplitudes)

result = np.hstack((reco_peak_info, amplitudes))
# Save results
np.savetxt('{}/x_denoised.txt'.format(output), result, fmt='%f', delimiter=' ')
