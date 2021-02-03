#!/usr/bin/python

# Use script by these arguments: input file, output file, model
# e.g:
# denoiser.py --noise signals_with_noise.npy --result denoised_signals.npy --model model_1.h5 --map mapping_info.npy
# input and output file format: NPY (numpy standart bit format)
import sys
import numpy as np
import argparse
from scipy.signal import hilbert
import glob
np.set_printoptions(threshold=np.nan)

def last_layer_out():
	activation_model = Model(inputs=autoencoder.layers[1].input, 
							outputs=autoencoder.layers[9].output)
	activation = activation_model.predict(test_trace)
	print(str(activation.shape)+'\t ACTIVATION SHAPE')
	print(activation)
	print(len(activation[0]))
	for j in range(len(activation[0])):
		temp_activation = activation[0][:][:]
		np.savetxt('out/layer4full.tab',temp_activation)

def find_max_n_shift(txt):
	#Take the trace with SNR bigger then threshold
	thres = 8
	data_dir = './traces/*'
	data_list = glob.glob(data_dir)
	for txt in data_list:
		noised_data = []
		f = open(txt)
		for line in f:
			noised_data.append(float(line))
		f.close()
		noised_data = np.array(noised_data)
		j_m = 2 * np.max(abs(hilbert(noised_data)))
		mean= np.mean(abs(hilbert(noised_data)))
		print('SNR = '+str(j_m/mean)+'\t'+txt)
		if j_m / mean > thres:
			noised_signals_shifted = (np.array(noised_data) / j_m) + 0.5
			noised_signals_shifted = np.array(noised_signals_shifted)
			trace = []
			trace.append(noised_signals_shifted)
			break
	return np.array(trace)
	
def read_file_n_shift(txt):
	#read trace from file
	noised_data = []
	f = open(txt)
	for line in f:
		noised_data.append(float(line))
	f.close()
	noised_data = np.array(noised_data)
	j_m = 2 * np.max(abs(hilbert(noised_data)))
	noised_signals_shifted = (np.array(noised_data) / j_m) + 0.5
	noised_signals_shifted = np.array(noised_signals_shifted)
	trace = []
	trace.append(noised_signals_shifted)
	return np.array(trace)

# Parse parameters and process it
parser = argparse.ArgumentParser()
#parser.add_argument('--no', '--noise', action='store', dest='noise')
parser.add_argument('-mo', '--model', action='store', dest='model')
args = parser.parse_args()

#noised_signals = args.noise
model = args.model
#result = args.result

from keras.models import load_model, Model
import matplotlib.pyplot as plt
import keras.backend as K
#import tensorflow as tf

autoencoder = load_model(model)
autoencoder.summary()

x_train = np.asarray([np.zeros(4096)])
x_train.fill(0)

#x_train = read_file_n_shift('traces/trace9.tab')
#x_train = find_max_n_shift('traces/trace9.tab')

#print(x_train)

shape_0 = x_train.shape[0]
shape_1 = x_train.shape[1]	
#print(shape_0, shape_1)
test_trace = np.reshape(x_train, (shape_0, shape_1, 1))

def filter_plotter_ext():
	timestep = 1.25*10**(-9)
	num = 0
	for layer in autoencoder.layers:
		if hasattr(layer,'filters'):
			print('name = '+str(layer.name)+' ==========================')
			print('filters = '+str(layer.filters))
			print('kernel_size = '+str(layer.kernel_size))
			print('input shape = '+str(layer.input_shape)+'\toutput shape = '+str(layer.output_shape))
			wsAndBs = np.array((layer.get_weights()[0]))
			print('initial shape = '+str(wsAndBs.shape))
#			print(len(wsAndBs[0]),len(wsAndBs[0][0]),len(wsAndBs[0][0]))
			if num != 0:
				shape_0 = wsAndBs.shape[0]
				shape_1 = wsAndBs.shape[2]
#				print(len(wsAndBs[0]),len(wsAndBs[0][0]))
#				wsAndBs = np.reshape(wsAndBs, (shape_0, shape_1))
				wsAndBs = (wsAndBs[:,0,:])
#				print('sliced shape = '+str(wsAndBs.shape))
				spec = np.array(np.real(np.fft.rfft(wsAndBs, axis=-2)))
				freq_list = np.array(np.fft.rfftfreq(len(spec), d=timestep))
				np.savetxt('out/layer'+str(num)+'.tab', spec**2)
			num+=1
			del wsAndBs
#			break

def compare_channels():
	timestep = 1.25*10**(-9)
	num = 0
	for layer in autoencoder.layers:
		if hasattr(layer,'filters'):
			print('name = '+str(layer.name)+' ==========================')
			print('filters = '+str(layer.filters))
			print('kernel_size = '+str(layer.kernel_size))
			print('input shape = '+str(layer.input_shape)+'\toutput shape = '+str(layer.output_shape))
			wsAndBs = np.array((layer.get_weights()[0]))
			if num != 0:
				wsAndBs1 = (wsAndBs[:,1,:])
				wsAndBs = (wsAndBs[:,0,:])
				np.savetxt('out/compare/layer'+str(num)+'ch0.tab', wsAndBs)
				np.savetxt('out/compare/layer'+str(num)+'ch1.tab', wsAndBs1)
			num+=1
			del wsAndBs
#			break

#==== SHOWS SIMILAR FILTERS ======
def filter_plotter():
	for num,name in enumerate([1,3,5,7,9]):	
#		activation_model = Model(inputs=autoencoder.layers[0].input, 
#								outputs=autoencoder.layers[name].output)
		wsAndBs = (autoencoder.layers[name].get_weights())
#		activation = activation_model.predict(test_trace)
#		print(str(wsAndBs.shape)+'\t weigths and biases shape')
#		print(wsAndBs)
#		print(activation)
		print(len(wsAndBs))
		for j in range(len(wsAndBs)):
			weigths = wsAndBs[0][j]
			np.savetxt('out/layer'+str(num)+'filter'+str(j)+'.tab',weigths)
#=================================
def full_plotter():
	np.savetxt('out/layer0filter0.tab',test_trace[0])
	for num,name in enumerate(range(1,10)):	
		activation_model = Model(inputs=autoencoder.layers[0].input, 
								outputs=autoencoder.layers[name].input)
		activation = np.array(activation_model.predict(test_trace))
#		print(str(activation.shape)+'\t ACTIVATION SHAPE')
#		print(activation)
		print(len(activation[0]))
		print('========================================================================')
		for j in range(len(activation[0][0])):
#			print(j)
			temp_activation = activation[0, ..., j]
#			print(temp_activation)
			np.savetxt('out/layer'+str(name)+'filter'+str(j)+'.tab',temp_activation)

#full_plotter()
filter_plotter_ext()
#compare_channels()
