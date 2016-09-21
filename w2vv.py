# Massimiliano Ruocco (ruocco@idi.ntnu.no)
# Implementation of the Word2VisualVec found in [1].
# [1] Jianfeng Dong, Xirong Li, Cees G. M. Snoek 2016. Word2VisualVec: Cross-Media Retrieval by Visual Feature Prediction
#		https://arxiv.org/pdf/1604.06838v1.pdf

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_yaml
# hyperparameters optimization https://github.com/maxpumperla/hyperas
from pandas import HDFStore,DataFrame
import pandas as pd
import glob, os
import keras

EPOCH = 10
BATCH_SIZE = 32
DROPOUT_RATE = 0.2 # tips for using dropout http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

# save/load models in keras http://machinelearningmastery.com/save-load-keras-deep-learning-models/
def load(yaml_filename, h5_weights_filename):
	# load YAML and create model
	yaml_file = open(yaml_filename, 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	# load weights into new model
	loaded_model.load_weights(h5_weights_filename)
	print("Loaded model from disk")
	return loaded_model;

#yaml_filename = "model.yaml"
#h5_weights_filename = "model.h5"
def save_model(model, yaml_filename, h5_weights_filename):
	# serialize model to YAML
	model_yaml = model.to_yaml()
	with open(yaml_filename, "w") as yaml_file:
	    yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights(h5_weights_filename)
	print("Saved model to disk")


def define_model():
	model = Sequential()
	# first layer h1
	model.add(Dense(800, input_dim=300))
	# second layer h2
	model.add(Dense(1500, activation="relu")) # specify dropout here
	model.add(Dropout(DROPOUT_RATE))
	# second layer h3
	model.add(Dense(2048, activation="relu"))
	# different objectives https://keras.io/objectives/
	adg = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
	model.compile(loss='mse', optimizer=adg)
	#model.compile(loss='mse', optimizer='rmsprop')
	return model

def define_model_rev():
	model = Sequential()
	# first layer h1
	model.add(Dense(1400, input_dim=2048))
	# second layer h2
	model.add(Dense(800, activation="relu")) # specify dropout here
	model.add(Dropout(DROPOUT_RATE))
	# second layer h3
	model.add(Dense(300, activation="relu"))
	# different objectives https://keras.io/objectives/
	adg = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
	model.compile(loss='mse', optimizer=adg)
	#model.compile(loss='mse', optimizer='rmsprop')
	return model

