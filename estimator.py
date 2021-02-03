#!/usr/bin/python

# arguments: true file, reco file, additional info, upsampling, mode
# e.g:
# estimator.py --true signals.npy --reco noise.npy --upsampling 16 --mode snr

import numpy as np
import pandas as pd
import argparse
from scipy.signal import hilbert


def estimator(true, reco, info):
    true_amplitude = get_amplitude(true)
    reco_amplitude = get_amplitude(reco)
    true_time = get_time(true)
    reco_time = get_time(reco)
    polarization = [0]*len(reco_time)

    result_table = pd.DataFrame(
        {#'event_id': event,
         'pol_id':polarization,
         'A_true': true_amplitude,
         'T_true': true_time,
         'A_reco': reco_amplitude,
         'T_reco': reco_time
         }
    )
    return result_table


def get_add_info(array):
    result_array = np.empty((0,2))
    for ind, elm in enumerate(array):
        if elm:
            event_stations_list = (np.asarray([[ind]*len(elm), elm]))
            result_array = np.append(result_array, event_stations_list.transpose(), axis=0)
    event, stations = result_array[:, 0], result_array[:, -1]
    return event, stations


def get_amplitude(array):
    amplitudes = [np.max(i) for i in array]
    amplitudes = np.array(amplitudes)
    return amplitudes


def get_time(array):
    array = np.load(array)
    time = [np.argmax(hilbert(i)) for i in array]
    time = np.array(time)
    time = time/ns
    return time


def snr_estimator(true, reco):
    true_peak_info = [get_peak_info(time, trace, window) for trace in true]
    reco_peak_info = [get_peak_info(time, trace, window) for trace in reco]
    snr_true = list(map(get_snr, np.hstack((true_peak_info, true))))
    snr_reco = list(map(get_snr, np.hstack((reco_peak_info, reco))))
    noise_true = list(map(get_noise, np.hstack((true_peak_info, true)))) 
    noise_reco = list(map(get_noise, np.hstack((reco_peak_info, reco))))
    result_table = pd.DataFrame(
        {
         'SNR_true': snr_true,
         'SNR_reco': snr_reco,
         'AMP_true': np.array(true_peak_info)[:, 0],
         'AMP_reco': np.array(reco_peak_info)[:, 0],
         'Time_true': np.array(true_peak_info)[:, 1],
         'Time_reco': np.array(reco_peak_info)[:, 1],
         'Noise_true': noise_true,
         'Noise_reco': noise_reco
         }
    )
    print ('SNR_true mean', result_table['SNR_true'].mean())
    print('SNR_reco mean', result_table['SNR_reco'].mean())
    print('AMP_reco > 40', sum(result_table['AMP_reco']) > 40)
    return result_table


def get_peak_info(time, trace, window):
    i = np.argmax(np.abs((trace-0.5))[window])
    return max(abs((trace-0.5))[window]), time[i] + time[window][0]


def get_snr(arr):
    peak_amp = arr[0]
    peak_position = arr[1]
    noise_window = np.where((time > peak_position + 50) | (time < peak_position - 50) | (time > 100) | (time < 1100))
    noise = np.std(arr[2:][noise_window])
    snr = peak_amp/noise
    return snr

def get_noise(arr):
    peak_amp = arr[0]
    peak_position = arr[1]
    noise_window = np.where((time > peak_position + 50) | (time < peak_position - 50) | (time > 100) | (time < 1100))
    noise = np.std(arr[2:][noise_window])
    return noise

# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--true', action='store', dest='true')
parser.add_argument('-r', '--reco', action='store', dest='reco')
parser.add_argument('-i', '--info', action='store', dest='info')
parser.add_argument('-up', '--upsampling', action='store', dest='upsampling')
parser.add_argument('-m', '--mode', action='store', dest='mode')
args = parser.parse_args()

true_signals = np.load(args.true)
reco_signals = np.load(args.reco)
print(true_signals.shape, reco_signals.shape)

if len(reco_signals.shape) == 3:
    reco_signals = np.reshape(reco_signals, (reco_signals.shape[0],reco_signals.shape[1]))

info = args.info
upsampling = int(args.upsampling)
mode = str(args.mode)

# translate len of trace to time
if upsampling == 16:
    ns = 0.325
    time = np.arange(len(true_signals[0])) * ns
else:
    ns = 1.25
    time = np.arange(len(true_signals[0])) * ns

window = np.where((time > 550) & (time < 800))

if mode == 'snr':
    result = snr_estimator(true_signals, reco_signals)
else:
    result = estimator(true_signals, reco_signals, info)

result.to_csv('./result.csv', index=None, sep=' ')

