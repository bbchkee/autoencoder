#!/usr/bin/python

# Use script by these arguments: input file, output file, model
# e.g:
# mappper.py --signal signals_with_noise.npy --noised denoised_signals.npy --denoised x_denoised.npy --map mapping_info.npy

def demapping(array1, array2, array3, mapping):
    x_test = []
    x_noisy = []
    x_denoised = []
    for i, j, k, z in zip(array1, array2, array3, mapping):
        x_test.append((i-0.5)*z[0])
        x_denoised.append((k-0.5)*z[0])
        x_noisy.append((j-0.5)*z[1])
    return x_test, x_noisy,  x_denoised


import sys
import numpy as np
import argparse


# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('-out', '--output', action='store', dest='output')
parser.add_argument('-xdn', '--xdenoised', action='store', dest='x_denoised')
parser.add_argument('-xt', '--xtest', action='store', dest='x_test')
parser.add_argument('-xtn', '--xtnoisy', action='store', dest='x_test_noisy')
parser.add_argument('-ma', '--map', action='store', dest='mapping_info')
args = parser.parse_args()


output = args.output
mapping_info = args.mapping_info
x_test = args.x_test
x_noisy = args.x_test_noisy
x_denoised = args.x_denoised

# Process data
mapping_info = np.load(mapping_info)
x_test = np.load(x_test)
x_noisy = np.load(x_noisy)
x_denoised = np.load(x_denoised)


# Demapping pipeline
x_test,x_noisy, x_denoised = demapping(x_test, x_noisy, x_denoised, mapping_info)

# Save results
np.save('{}/x_denoised_demap.npy'.format(output), x_denoised)
np.save('{}/x_test_demap.npy'.format(output), x_test)
np.save('{}/x_test_noisy_demap.npy'.format(output), x_noisy)
