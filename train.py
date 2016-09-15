# Massimiliano Ruocco (ruocco@idi.ntnu.no)
#
# Main file for training the model from one/set of h5 file with 'data'

import w2vv
import glob,os
from pandas import HDFStore,DataFrame
import numpy as np
import sys
import getopt
#from keras.callbacks import ModelCheckpoint

EPOCH = 10
BATCH_SIZE = 32
DROPOUT_RATE = 0.2 # tips for using dropout http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

def train(inputfolder, outputfolder):
	labels = []
	input = []
	os.chdir(inputfolder)
	count = 0
	for file in glob.glob("*.h5"):
		input_h5 = inputfolder + file
		print(input_h5)
		hdf = HDFStore(input_h5)
		labels = labels + hdf.data.img_feats.tolist()
		input = input + hdf.data.txt_feats.tolist()
		hdf.close()
		count = count + 1
		if count == 1:
			break
	print(len(labels))
	model = w2vv.define_model()
	# checkpoint
	#filepath="weights.best.hdf5"
	#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#callbacks_list = [checkpoint]
	# Fit the model
	#model.fit(np.stack(input), np.stack(labels), validation_split=0.33, nb_epoch=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=0)
	model.fit(np.stack(input), np.stack(labels), nb_epoch=EPOCH, batch_size=BATCH_SIZE)
	yaml_filename = "outVw2v.yaml"
	h5_weights_filename = "outVw2v_weights.h5"
	os.chdir(outputfolder)
	w2vv.save_model(model, yaml_filename, h5_weights_filename)

def main(argv):
	inputfolder = ''
	outputfolder = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["if=","of="])
	except getopt.GetoptError:
		print 'test.py -i <inputfolder> -o <outputfolder>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfolder> -o <outputfolder>'
			sys.exit()
		elif opt in ("-i", "--if"):
			inputfolder = arg
		elif opt in ("-o", "--of"):
			outputfolder = arg
	print 'Input folder is "', inputfolder
	print 'Output folder is "', outputfolder
	train(inputfolder, outputfolder)

if __name__ == "__main__":
	main(sys.argv[1:])



