#!/usr/bin/env python
# coding: utf-8

# # Malafeev CNN reference method
# 
# Malafeev, A., Hertig-Godeschalk, A., Schreier, D. R., Skorucak, J., Mathis, J., & Achermann, P. (2021). Automatic Detection of Microsleep Episodes With Deep Learning. Frontiers in Neuroscience, 15, 564098. https://doi.org/10.3389/fnins.2021.564098
# 
# This is a reference method with slightly modified implemenations

# ## Build model

# In[24]:


# Build 16-CNN Model

import os

from keras import backend as K
from keras import optimizers

from keras.callbacks import History 
from keras.layers import concatenate
from keras.layers import Layer,Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import  GRU, Bidirectional
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D, SpatialDropout2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import np_utils


def build_model(data_dim, n_channels, n_cl):
	eeg_channels = 1
	act_conv = 'relu'
	init_conv = 'glorot_normal'
	dp_conv = 0.3
	def cnn_block(input_shape):
		input = Input(shape=input_shape)
		x = GaussianNoise(0.0005)(input)
		x = Conv2D(32, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
		x = BatchNormalization()(x)
		x = Activation(act_conv)(x)
		x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		
		
		x = Conv2D(64, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
		x = BatchNormalization()(x)
		x = Activation(act_conv)(x)
		x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		for i in range(4):
			x = Conv2D(128, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
			x = BatchNormalization()(x)
			x = Activation(act_conv)(x)
			x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		for i in range(6):
			x = Conv2D(256, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
			x = BatchNormalization()(x)
			x = Activation(act_conv)(x)
			x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
		flatten1 = Flatten()(x)
		cnn_eeg = Model(inputs=input, outputs=flatten1)
		return cnn_eeg
		
	hidden_units1  = 256
	dp_dense = 0.5

	eeg_channels = 1
	eog_channels = 2

	input_eeg = Input(shape=( data_dim, 1,  3))
	cnn_eeg = cnn_block(( data_dim, 1, 3))
	x_eeg = cnn_eeg(input_eeg)
	x = BatchNormalization()(x_eeg)
	x = Dropout(dp_dense)(x)
	x =  Dense(units=hidden_units1, activation=act_conv, kernel_initializer=init_conv)(x)
	x = BatchNormalization()(x)
	x = Dropout(dp_dense)(x)

	predictions = Dense(units=n_cl, activation='softmax', kernel_initializer=init_conv)(x)

	model = Model(inputs=[input_eeg] , outputs=[predictions])
	return [cnn_eeg, model]

cnn, model = build_model(3200, 3, 2)
model.summary()


# ## Load and pre-process data

# In[25]:


import json

# Load some random subjects from train splits
with open("./splits/skorucack_splits.json") as f:
    splits = json.loads(f.read())


# In[26]:


import numpy as np
from scipy.io import loadmat
from keras.utils.np_utils import to_categorical

class MalafeevStudy(object):

    fs = 200
    win_len = int(16 * fs)
    pad_len = int(win_len / 2)

    def __init__(self, signal_file, target_file):
        
        tmp = loadmat(signal_file, struct_as_record=False, squeeze_me=True)
        
        pre_process = lambda x: np.clip(x / 100, -1, 1)
        padding = lambda x: np.pad(x, pad_width=(self.pad_len, self.pad_len), mode="constant", constant_values = (0, 0))

        self.E1 = np.expand_dims(padding(pre_process(tmp['Data'].E1)),-1)
        self.E2 = np.expand_dims(padding(pre_process(tmp['Data'].E2)), -1)
        O1 = np.expand_dims(padding(pre_process(tmp['Data'].eeg_O1)), -1)
        O2 = np.expand_dims(padding(pre_process(tmp['Data'].eeg_O2)), -1)
        self.EEG = [O1, O2]

        targets = loadmat(target_file)['x']
        targets[targets!=1] = 0
        self.y = to_categorical(targets, num_classes=2) 

        self.num_win = len(self.y)


    def get_window_by_idx(self, i, c):
        
        
        start = int(i*self.fs)
        end = int(start+self.win_len)

        x = np.concatenate([self.E1[start:end], self.E2[start:end], self.EEG[c][start:end]],axis=1)
        x = np.expand_dims(x, -2)

        y = self.y[i]

        return x, y

    def get_sample_idx(self, study_idx):

        sample_idx = np.empty([self.num_win * 2, 3], dtype=int)
        for c in range(2):
            for i in range(self.num_win):
                ix=int(c*self.num_win+i)
                sample_idx[ix,...]=(study_idx, i, c)
        return sample_idx            


# In[27]:


import tensorflow as tf

def flatten(l):
    return [item for sublist in l for item in sublist]

np.random.seed(42)

class Generator(tf.keras.utils.Sequence):

    def __init__(self, split, batch_size):
        self.studies = [MalafeevStudy(signal_file=f"Matlab/data/{f}", target_file=f"edf_data/{f.replace('.mat','_status.mat')}") for f in split]
        self.indices = flatten([tmp.get_sample_idx(i) for i, tmp in enumerate(self.studies)])
        self.batch_size = batch_size
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        
        batch_x = np.empty([self.batch_size, self.studies[0].win_len, 1, 3])
        batch_y = np.empty([self.batch_size, 2])

        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        for i, idx in enumerate(idxs):
            x, y = self.get_study_windows_by_idx(idx)
            batch_x[i,...] = x
            batch_y[i,...] = y
        return batch_x, batch_y

    def get_study_windows_by_idx(self, index):

        return self.studies[index[0]].get_window_by_idx(index[1], index[2])

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

train_data = Generator(split=splits['train'], batch_size=200)


# ## Train model

# In[35]:


from keras import backend

fs = 200
win_sec = 16
win_len = win_sec * fs
n_classes = 2
n_channels = 3

#ordering = 'channels_first'
#backend.set_image_data_format(ordering)

_, model = build_model(win_len, n_channels, n_classes)


# In[36]:


from sklearn.utils import compute_class_weight

# Compute class weights
y = flatten([x.y[...,1] for x in train_data.studies])
cls = np.arange(n_classes)
clw = compute_class_weight(class_weight="balanced", classes=cls, y=y)
class_weights = {0: clw[0], 1: clw[1]}


# In[37]:


from tensorflow import keras


model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.002),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=keras.metrics.CategoricalAccuracy())


with tf.device("/device:GPU:0"):
    history = model.fit(train_data, 
                        class_weight=class_weights,
                        epochs=3)


# ## Save results

# In[ ]:


import pickle
model.save_weights("CNN_weights.h5")

with open('history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# ## Evaluate test data

# # In[7]:


# import os

# # Load model and load weights
# folder = "malafeev"
# weight_file = os.path.join(folder,"CNN_weights.h5")

# _, model = build_model(win_len, n_channels=n_channels, n_cl=n_classes)
# model.load_weights(weight_file)


# # In[18]:


# def predict_study(mdl, study: MalafeevStudy):

#     n_classes = mdl.output_shape[-1]
#     n_channels = mdl.input_shape[-1]

#     ch_x = np.empty([study.num_win, study.win_len, 1, n_channels])
#     ch_y = np.empty([study.num_win])
#     study_preds = np.empty([study.num_win, n_classes, n_channels])

#     idxs = study.get_sample_idx(0)

#     for ch in range(n_channels):
        
#         ch_idx = idxs[np.where(idxs[:,2]==ch)[0],1:3]
        
#         for i, idx in enumerate(ch_idx):
#             x, y = study.get_window_by_idx(*idx)
#             ch_x[i,...] = x
#             ch_y[i,...] = y[-1]

#         study_preds[...,ch] = mdl.predict_on_batch(ch_x)

#     return np.mean(study_preds, axis=2), ch_y    


# # In[20]:


# y_pred = []
# y_true = []
# ids = []

# for fi in splits['test']:
#     print(f"Predicting study: {fi}")
#     ids.append(fi)

#     sig_file = f"Matlab/data/{fi}.mat"
#     y_file = f"edf_data/{fi}_status.mat"
#     study = MalafeevStudy(signal_file=sig_file, target_file=y_file)

#     study_prob, study_y = predict_study(model, study)
#     y_pred.append(np.argmax(study_prob, axis=1))
#     y_true.append(study_y)


# # In[22]:


# from sklearn.metrics import recall_score, precision_score, f1_score, cohen_kappa_score, confusion_matrix

# y_true=[(y==1)*1 for y in y_true]
# y_hat = np.concatenate(y_pred)
# y = np.concatenate(y_true)==1

# recall = recall_score(y, y_hat)
# precision = precision_score(y, y_hat)
# f1 = f1_score(y, y_hat)
# kappa = cohen_kappa_score(y, y_hat)

# print(f"Recall:\t\t{recall:.2f}")
# print(f"Precision:\t{precision:.2f}")
# print(f"F1-Score:\t{f1:.2f}")
# print(f"Cohen's kappa:\t{kappa:.2f}")


# # In[23]:


# from scipy.io import savemat

# ids = splits['test']
# out = {"id": ids,
#     "yHat": y_pred,
#     "yTrue": y_true}

# savemat("Matlab/malafeev.mat", mdict=out)

