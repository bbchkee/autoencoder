#!/usr/bin/python

# Used for convert from ADST to numpy binary format only.
# Arguments: adstfile without noise, adstfile with noise
# e.g:
# converter.py --signal ADST-nonoise-nocorr.root --noise ADST-noise-nocorr.root

import ROOT
import numpy as np
from pyik.adst import RecEventProvider, RecEventHandler, MICRO, FULL, ROOT
import argparse


def converter(adstfile1, adstfile2):
    """convert adst to numpy"""
    # Create array with no-noise data
    signal_list = []
    file_name_nonoise = adstfile1
    adstHandler_nonoise = RecEventHandler(file_name_nonoise, mode=MICRO)
    adstEvent_nonoise = ROOT.RecEvent()
    adstHandler_nonoise.SetBuffers(adstEvent_nonoise)
    nEvents = adstHandler_nonoise.GetNEvents()
    for event in xrange(nEvents):
        adstHandler_nonoise.ReadEvent(event)
        for station in adstEvent_nonoise.GetRdEvent().GetRdStationVector():
            if not station.HasPulse(): continue
            trace = station.GetRdTrace(0)
            samples = trace.GetTimeTrace()
            signal = [x for x in samples]
            signal_list.append(signal)

    signal_list = np.array(signal_list)
    np.save('signals_row.npy', signal_list)
    del signal_list

    # Create list for reading noise data by index
    index_list = []
    for event in xrange(nEvents):
        adstHandler_nonoise.ReadEvent(event)
        station_list = []
        for st_index, station_nonoise in enumerate(adstEvent_nonoise.GetRdEvent().GetRdStationVector()):
            if not station_nonoise.HasPulse(): continue
            station_list.append(st_index)
        index_list.append(station_list)
    index_list = np.array(index_list)
    del adstEvent_nonoise

    # Create array with noise data
    file_name_noise = adstfile2
    adstHandler_noise = RecEventHandler(file_name_noise, mode=MICRO)
    adstEvent_noise = ROOT.RecEvent()
    adstHandler_noise.SetBuffers(adstEvent_noise)
    nEvents = adstHandler_noise.GetNEvents()
    noise_list = []
    for index, event in enumerate(xrange(nEvents)):
        adstHandler_noise.ReadEvent(event)
        for station_noise in np.array(adstEvent_noise.GetRdEvent().GetRdStationVector())[index_list[index]]:
            trace_noise = station_noise.GetRdTrace(0)
            samples_noise = trace_noise.GetTimeTrace()
            signal_noise = [x for x in samples_noise]
            noise_list.append(signal_noise)

    noise_list = np.array(noise_list)
    np.save('noise_row.npy', noise_list)
    del noise_list, adstEvent_noise, index_list
    return True


# Parse parameters and process it
parser = argparse.ArgumentParser()
parser.add_argument('--wo', '--signal', action='store', dest='signal')
parser.add_argument('-wn', '--noise', action='store', dest='noise')

args = parser.parse_args()

adst_signal = args.signal
adst_noise = args.noise


# Convert adst to numpy
converter(adst_signal, adst_noise)

