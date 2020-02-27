##########################################
# Krishna Dave
##########################################

# Libraries 
import pickle
import sys
import scipy.io as sio
import copy
from numpy import array
from numpy import hstack
import numpy as np
import pandas as pd
from frame import Frame
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import RepeatVector
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
sys.path.append('/Users/krishnadave/Desktop/DPGP/')

##############################
# Files to work with
file1 = 'argo_MixtureModel_2600_2640_1210_1250'
file2 = 'argo_MixtureModel_2740_2780_1330_1370'
file3 = 'argo_MixtureModel_2780_2810_1360_1390'
file4 = 'argo_MixtureModel_2570_2600_1180_1210'
file5 = 'argo_MixtureModel_2640_2670_1240_1270'
file6 = 'argo_MixtureModel_2670_2710_1270_1310'
file7 = 'argo_MixtureModel_2710_2740_1300_1330'

file_names = [file2, file3, file4, file5, file6, file7]
#file_names = [file4]
##############################

TRAIN_SPLIT = 0.7

def read_dataset(file_name):
	with open(file_name, 'rb') as infile:
		data = pickle.load(infile)
		infile.close()
		return data

def prepare_covMat_data(file_name1):

	read_data = read_dataset(file_name1)
	motion_patterns = read_data.b

	# need to write a helper function for this
	xmin, xmax, ymin, ymax = get_minmax(file_name1)

	WX, WY, ux_pos, uy_pos = covariance_matrix_calc(read_data, motion_patterns, xmin, xmax, ymin, ymax)

	stacked_columns = np.column_stack([WX, WY, ux_pos, uy_pos])
	dataset = pd.DataFrame(stacked_columns, columns=['positionX', 'positionY', 'errorX', 'errorY']).values

	data_mean = dataset[:round(TRAIN_SPLIT * len(dataset))].mean(axis=0)
	data_std = dataset[:round(TRAIN_SPLIT * len(dataset))].std(axis=0)

	dataset = (dataset - data_mean) / data_std

	return dataset


def covariance_matrix_calc(mix_model, motion_patterns, xmin, xmax, ymin, ymax):
	# Takes in the dictionary patt_frame_dict for pattern - frames (key - value pairs) 
	#                         motion patterns  (list with motion pattern info)	
	# Returns the covariance matrix calculated for each pattern based on their corresponding frames
 
	# would the mix model consist of frames for all patterns, or all data?
	
	# The aim of the pattern frame dictionary
	# for each motion pattern, how do I calculate covariance matrix based on the corresponding frames
	for i in range(0, len(motion_patterns)):
		# for pattern in motion_patterns:
		pattern_num = i
		#print("mixmodel.partition", mix_model.partition) # number of frames assigned to the pattern
		pattern_num = np.argmax(np.array(mix_model.partition))
		rate = np.array(mix_model.partition)[i]/mix_model.n
		frame_pattern_ink = mix_model.frame_ink(pattern_num, 0, True)
		# construct mesh frame
		x = np.linspace(xmin, xmax, 31)
		y = np.linspace(ymin, ymax, 31)
		[WX,WY] = np.meshgrid(x, y)
		WX = np.reshape(WX, (-1, 1))
		WY = np.reshape(WY, (-1, 1))
		frame_field = Frame(WX.ravel(), WY.ravel(), np.zeros(len(WX)), np.zeros(len(WX)))
		#get posterior
		ux_pos, uy_pos, covx_pos, covy_pos = mix_model.b[pattern_num].GP_posterior(frame_field, frame_pattern_ink, True)
	return [WX, WY, ux_pos, uy_pos]

# Helper for LSTM
def get_minmax(file_name):
	splits = file_name.split("_")
	xmin = int(splits[-4])
	xmax = int(splits[-3])
	ymin = int(splits[-2])
	ymax = int(splits[-1])
	return xmin, xmax, ymin, ymax

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  fig = plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.xlabel('Number of Epochs')
  plt.ylabel('MSE Loss')
  fig.savefig("SGD_plots/" + title + "_plot.jpg")
  #plt.show()


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

if __name__ == "__main__":
	for file_name in file_names:
		dataset = prepare_covMat_data(file_name)

		#print(dataset[0:7])
		#print(file_name, len(dataset), len(dataset[0]))
		
		print(dataset[:, 0:4])
		
		# HYPER PARAMETERS
		past_history = 5
		future_target = 2
		BATCH_SIZE = 50
		BUFFER_SIZE = 10000
		EPOCHS = 40
		EVALUATION_INTERVAL = 200 
		VALIDATION_STEPS = 50
		STEP = 1

		# SPLITTING THE DATA SET - TRAINING AND VALIDATION
		x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0:4], 0,
	                                                   round(TRAIN_SPLIT * len(dataset)), past_history,
	                                                   future_target, STEP,
	                                                   single_step=False)

		x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0:4],
	                                               round(TRAIN_SPLIT * len(dataset)), None, past_history,
	                                               future_target, STEP,
	                                               single_step=False)


		# BATCHING AND SHUFFLING
		train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
		train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

		val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
		val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

		# MODEL SETUP
		num_features = 4 # remove hardcoded value 
		model = Sequential()
		model.add(LSTM(2, activation='relu', input_shape=(past_history, num_features)))
		model.add(RepeatVector(future_target))
		model.add(LSTM(2, activation='relu', return_sequences=True))
		model.add(TimeDistributed(Dense(num_features)))
		model.compile(optimizer='sgd', loss='mse') # adam

		multi_step_history = model.fit(train_data_single, epochs=EPOCHS,
	                                            steps_per_epoch=EVALUATION_INTERVAL,
	                                            validation_data=val_data_single,
	                                            validation_steps=VALIDATION_STEPS)

		plot_train_history(multi_step_history, file_name + " Loss")
		



