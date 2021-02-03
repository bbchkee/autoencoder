#!/usr/bin/python

# Use script by these arguments: input file
# e.g:
# plotter.py --input result.csv

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import argparse
import os


def plot_hist(inp_file):
    data = pd.read_csv(inp_file, sep=" ")
    true_time = data['T_true']
    reco_time = data['T_reco']
    true_amplitude = data['A_true']
    reco_amplitude = data['A_reco']
    time_diff = true_time - reco_time
    amplitude_diff = true_amplitude - reco_amplitude
    plt.hist(time_diff)
    plt.savefig('time_diff.pdf')
    plt.clf()
    plt.hist(amplitude_diff)
    plt.savefig('amplitude_diff.pdf')
    return True

def plot_snr_hist(inp_file, plot_all = False):
    os.makedirs('result_hist')
    data = pd.read_csv(inp_file, sep=" ")
    snr_true = data['SNR_true']
    snr_reco = data['SNR_reco']
    noise_true = data['Noise_true']
    noise_reoc = data['Noise_reco']
    true_amplitude = data['AMP_true']
    reco_amplitude = data['AMP_reco']
    data['SNR_diff'] = data['SNR_reco'] - data['SNR_true']
    time_true = data['Time_true']
    time_reco = data['Time_reco']
    # time scatter
    plt.scatter(time_true, time_reco)
    plt.xlabel('True peak pos.')
    plt.ylabel('Reco peak pos.')
    plt.savefig('result_hist/time_scatter.pdf')
    plt.clf()
    # amp scatter
    plt.scatter(true_amplitude, reco_amplitude)
    plt.xlabel('True amplitude')
    plt.ylabel('Reco amplitude')
    plt.savefig('result_hist/amp_scatter.pdf')
    plt.clf()
    # amp hist
    amp_true_mean = data['AMP_true'].mean()
    amp_reco_mean = data['AMP_reco'].mean()
    plt.hist(data['AMP_true'], color='blue', label = ('Initial ' + str(amp_true_mean)))
    plt.hist(data['AMP_reco'], color='green', alpha = 0.7, label = ('Reco ' + str(amp_reco_mean)))
    plt.legend(loc = 'upper right')
    plt.savefig('result_hist/amp_hist.pdf')
    plt.clf()
    # snr hist
    snr_true_mean = data['SNR_true'].mean()
    snr_reco_mean = data['SNR_reco'].mean()
    plt.hist(data['SNR_true'], color='blue', label = ('Initial ' + str(snr_true_mean)))
    plt.hist(data['SNR_reco'], color='green', alpha = 0.7, label = ('Reco ' + str(snr_reco_mean)))
    plt.legend(loc = 'upper right')
    plt.savefig('result_hist/snr_hist.pdf')
    plt.clf()
    # best snr example
    index = data.index[data['SNR_diff'] == data['SNR_diff'].max()].tolist()[0]
    x = np.arange(x_test[0][100:4086].shape[0])
    plt.figure(figsize=(20, 10))
    plt.plot(x, x_test_noisy[index][100:4086], color = 'red')
    plt.plot(x, x_test[index][100:4086], color='green')
    plt.plot(x, x_denoised[index][100:4086], color='blue')
    plt.savefig('result_hist/best_example.pdf')
    plt.clf()
    #best with rescall
    index = data.index[data['SNR_diff'] == data['SNR_diff'].max()].tolist()[0]
    x = np.arange(x_test[0][100:4086].shape[0]) 
    rescalling = float(max(abs(hilbert(x_test_noisy[index][100:4086]))))/max(abs(hilbert(x_denoised[index][100:4086])))
    plt.figure(figsize=(20,10))
    plt.plot(x, x_test_noisy[index][100:4086], color = 'red')
    plt.plot(x, x_denoised[index][100:4086]*rescalling, color='blue')
    plt.savefig('result_hist/best_example_rescalling.pdf')
    plt.clf()
    # noise hist
    noise_true_mean = data['Noise_true'].mean()
    noise_reco_mean = data['Noise_reco'].mean()
    plt.hist(data['Noise_true'], color='blue', label = ('Initial ' + str(noise_true_mean))) 
    plt.hist(data['Noise_reco'], color='green', label = ('Reco ' + str(noise_reco_mean)))
    plt.legend(loc = 'upper right')
    plt.savefig('result_hist/noise_hist.pdf')
    plt.clf()
    # amp_reco vs amp_true
    plt.hist(true_amplitude, color='blue')
    plt.hist(reco_amplitude[reco_amplitude > 40], color='green')
    plt.savefig('result_hist/amp_true_reco.pdf')
    plt.clf()
    if plot_all == True:
        x = np.arange(x_test[0][100:4086].shape[0])
        for i in range(x_test.shape[0]):
            plt.plot(x, x_test_noisy[i][100:4086], color = 'red')
            plt.plot(x, x_test[i][100:4086], color = 'blue')
            plt.plot(x, x_denoised[i][100:4086], color = 'green')
            plt.savefig('result_hist/pic_{}.pdf'.format(i))
            plt.clf() 
    return True

# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', dest='input_file')
parser.add_argument('-xdn', '--xdenoised', action='store', dest='x_denoised')
parser.add_argument('-xt', '--xtest', action='store', dest='x_test')
parser.add_argument('-xtn', '--xtnoisy', action='store', dest='x_test_noisy')
parser.add_argument('-m', '--mode', action='store', dest='mode')
args = parser.parse_args()

input_file = args.input_file
x_test = args.x_test
x_test_noisy = args.x_test_noisy
x_denoised = args.x_denoised
mode = args.mode

x_test = np.load(x_test)
x_test_noisy = np.load(x_test_noisy)
x_denoised = np.load(x_denoised)

if mode == 'all':
    plot_snr_hist(input_file, plot_all = True)
else:
    plot_hist(input_file, plot_all = False)
