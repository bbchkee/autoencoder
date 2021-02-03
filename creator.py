#!/usr/bin/python

# Used for cut or create dataset (more in readme.txt)
# Arguments: file without noise, file with noise, center, window, upsampling (4 or 16 only).
# e.g:
# creator.py --signal signal.npy --noise noise.npy --center 2048 --window 1024

import numpy as np
import argparse


def processing(array1, array2, center, window, augmentation):
    """process data"""
    size = (window)/2
    center = center
    unc = [0]
    unc.extend([augmentation, -augmentation])
    signals = []
    noised_signals = []
    for i, j in zip(array1, array2):
        shift = np.random.choice(unc)
        start_position, end_position = center + shift - size, center + shift + size
        start_position,end_position = int(start_position), int(end_position)
        signals.append(i[start_position:end_position])
        noised_signals.append(j[start_position:end_position])
    return signals, noised_signals


# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('--wo', '--signal', action='store', dest='signal')
parser.add_argument('-wn', '--noise', action='store', dest='noise')
argument = parser.add_argument('-c', '--center', action='store', dest='center')
add_argument = parser.add_argument('-w', '--window', action='store', dest='window')
parser.add_argument('-up', '--upsampling', action='store', dest='up')
args = parser.parse_args()

input_signals = args.signal
input_noised_signals = args.noise
center = int(args.center)
window = int(args.window)

if not bool(args.upsampling):
    aug = 0
else:
    upsampling = int(args.upsampling)
    if upsampling == 16:
        aug = 150
    else:
        aug = 50

signals_file, noised_signals_file = processing(input_signals, input_noised_signals, center, window, aug)
np.save('signals.npy', signals_file)
np.save('noised_signals.npy', noised_signals_file)
