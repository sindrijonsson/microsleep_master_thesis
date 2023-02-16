#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import os
import usleep
import typing

from IPython.display import clear_output


# In[82]:


# ARGS

class ARGS(object):
    def __init__(self, file, data_per_prediction: int = 128):
        self.f = os.path.abspath(file) if not os.path.isabs(file) else file
        self.o = self.f.replace(".edf",".npy")
        self.logging_out_path = self.f.replace(".edf",".log")
        
        self.auto_channel_grouping =  ['EOG', 'EEG']
        self.auto_reference_types  =  None
        self.channels              =  ['O1-M2==EEG', 'O2-M1==EEG', 'E1-M1==EOG', 'E2-M1==EOG']
        self.data_per_prediction   =  128
        self.force_gpus            =  ''
        self.header_file_name      =  None
        self.model                 =  'u-sleep:1.0'
        self.no_argmax             =  True
        self.num_gpus              =  0
        self.overwrite             =  True
        self.project_dir           =  usleep.get_model_path(model_name=self.model.split(":")[0], model_version=self.model.split(":")[-1])
        self.strip_func            =  'trim_psg_trailing'
        self.weights_file_name     =  None


# In[83]:


from utime import Defaults
from utime.hyperparameters import YAMLHParams

# Load arguments and hyperparamets
args = ARGS(file="edf_data/9JQY.edf", data_per_prediction=128)
hparams = YAMLHParams(Defaults.get_hparams_path(args.project_dir), no_version_control=True)


# In[84]:


from psg_utils.dataset.sleep_study import SleepStudy
from utime.bin.predict_one import get_sleep_study

def get_and_load_study(file, args: ARGS, hparams: YAMLHParams) -> SleepStudy:

    # Get the sleep study
    print(f"Loading and pre-processing PSG file {file}...")
    hparams['prediction_params']['channels'] = args.channels
    hparams['prediction_params']['strip_func']['strip_func_str'] = args.strip_func

    study, channel_groups = get_sleep_study(psg_path=file,
                                            header_file_name=args.header_file_name,
                                            auto_channel_grouping=args.auto_channel_grouping,
                                            auto_reference_types=args.auto_reference_types,
                                            **hparams['prediction_params'])
    
    study.channel_groups = channel_groups

    return study


# In[85]:


from utime.bin.evaluate import get_and_load_model, get_and_load_one_shot_model
from keras import Model
from keras.layers import Input

def init_muSleep(args: ARGS, hparams: YAMLHParams, study: SleepStudy = None, freeze_base = True) -> Model:

    # Load pre-trained U-Sleep Model and attach new head
    if study is None:
        model = get_and_load_model(
            project_dir=args.project_dir,
            hparams=hparams,
            weights_file_name=hparams.get('weights_file_name')
        )
    else:
        model = get_and_load_one_shot_model(
                                            n_periods=study.n_periods,
                                            project_dir=args.project_dir,
                                            hparams=hparams,
                                            weights_file_name=hparams.get('weights_file_name')
                                            )
    clear_output(wait=False)    # Removing glorot intitialization warning...

    # Freeze base layers
    if freeze_base:
        for layer in model.layers[:-7]:
            layer.trainable = False


    # Extract base from pre-trained model (remove last )
    base = model.layers[-5].output

    # Create new head with base model as input with a 2-class problem
    head=model.create_seq_modeling(in_=base,
                                input_dims=model.input_dims,
                                data_per_period=args.data_per_prediction,
                                n_periods=model.n_periods,
                                n_classes=2,
                                transition_window=model.transition_window,
                                activation=model.activation,
                                regularizer=None)


    return Model(inputs=model.input, outputs = head, name = "mU-Sleep")


# In[86]:


# Function to predict on study (note the model given must be loaded by study)

def predict_on_study(model: Model, study: SleepStudy) -> np.array:

    print(f"Predicting on study {study.psg_file_path}")
    psg = np.expand_dims(study.get_all_periods(),0)
    subset = psg[...,study.channel_groups[0].channel_indices]
    preds=model.predict_on_batch(subset)
    return preds.reshape(-1, preds.shape[-1])
    


# In[87]:


# def my_predict_study(study, model, channel_groups, no_argmax):
#     psg = np.expand_dims(study.get_all_periods(), 0)
#     pred = np.empty([len(channel_groups), study.n_periods*30, model.n_classes])
#     for i, channel_group in enumerate(channel_groups):
#         # Get PSG for particular group
#         psg_subset = psg[..., tuple(channel_group.channel_indices)]
#         pred_i = model.predict_on_batch(psg_subset)
#         pred[i,...] = pred_i.reshape(-1, pred.shape[-1])
#     return pred

# _hparams = YAMLHParams(Defaults.get_hparams_path(args.project_dir), no_version_control=True)
# _hparams["build"]["data_per_prediction"] = args.data_per_prediction

# from scipy.io import savemat

# for _s in dev_studies:
#     _model = get_and_load_one_shot_model(
#                                         n_periods = _s.n_periods,
#                                         project_dir=args.project_dir,
#                                         hparams=_hparams,
#                                         weights_file_name=_hparams.get('weights_file_name'))
#     clear_output(wait=False)

#     p=my_predict_study(_s, _model, _s.channel_groups, True)
#     n = _s.psg_file_path.split('\\')[-1].replace('.edf','.mat')
#     savemat(f"probs/{n}", mdict={"prbobs": p})


# # Tranfer learning
# Steps:
# 1. Create muSleep model with non-trainable base model (base U-Sleep with new head)
# 2. Get a list of SleepStudy objects to train on
# 3. Initialize tf.Sequence object with list of SleepStudies with the following ´__getitem__´ functionality
#     1. Select random sleep study (or balanced?)
#     2. Select random period with margin (limit the randomness within margins)
#     3. Select random channels
#     4. Extract psg based on the above
#     5. Extract y (labels) with the same indexing as above
#     6. Repeat for each batch

# ## Initialize model to train
# 

# In[ ]:





# In[88]:


# Initialize model to train
hparams = YAMLHParams(Defaults.get_hparams_path(args.project_dir), no_version_control=True)
hparams['build']['batch_shape'] = [64, 19, 3840, 2]

# model = init_muSleep(args, hparams)

base = get_and_load_model(
            project_dir=args.project_dir,
            hparams=hparams,
            weights_file_name=hparams.get('weights_file_name')
        )
clear_output(wait=False)    # Removing glorot intitialization warning...

# Freeze base layers
base.trainable = False

base.layers[-5].trainable = True

# Extract base from pre-trained model (remove last )
inter_base = Model(inputs=base.input, outputs=base.layers[-5].output, name="uSleep_base")
inter_out = inter_base(inputs=base.input, training=False)

# Create new head with base model as input with a 2-class problem
head=base.create_seq_modeling(in_=inter_out,
                            input_dims = base.input_dims,
                            data_per_period=args.data_per_prediction,
                            n_periods=base.n_periods,
                            n_classes=2,
                            transition_window=base.transition_window,
                            activation=base.activation,
                            regularizer=None)


model = Model(inputs=base.input, outputs = head, name = "mU-Sleep")

print(f"Input shape: {model.input.shape}")
print(f"Output shape: {model.output.shape}")
print(f"Model trainable: {model.trainable}")
_=[print(f"\t {x.name}: ({x.shape})") for x in model.trainable_weights]
model.summary()


# In[89]:


import json

# Load some random subjects from train splits
with open("./splits/skorucack_splits.json") as f:
    splits = json.loads(f.read())

dev_studies = [get_and_load_study(f"edf_data/{x}.edf", args, hparams) for x in splits['train']]
test_studies = [get_and_load_study(f"edf_data/{x}.edf", args, hparams) for x in splits['test']]
clear_output(wait=False)


# In[90]:


# Let's add some random labels to the studies (for now)
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from scipy.io import loadmat

for study in dev_studies:
    psg_shape = study.get_all_periods().shape
    _y = loadmat(study.psg_file_path.replace(".edf",".mat"), squeeze_me=True)['x']
    _y[np.where(_y != 1)[0]] = 0
    _y = to_categorical(_y, num_classes = 2)
    # y = np.random.randint(0, 2, size = [psg_shape[0], int(psg_shape[1] / study.sample_rate), 1])
    # y=np.arange(1,study.get_psg_shape()[0]/study.sample_rate + 1)
    # y=np.reshape(y, [psg_shape[0], int(psg_shape[1] / study.sample_rate) , 1])
    study.y = np.reshape(_y, [psg_shape[0], int(psg_shape[1] / study.sample_rate) , 2])

# Let's add some random labels to the studies (for now)
for study in test_studies:
    psg_shape = study.get_all_periods().shape
    _y = loadmat(study.psg_file_path.replace(".edf",".mat"), squeeze_me=True)['x']
    _y[np.where(_y != 1)[0]] = 0
    _y = to_categorical(_y, num_classes = 2)
    # y = np.random.randint(0, 2, size = [psg_shape[0], int(psg_shape[1] / study.sample_rate), 1])
    # y=np.arange(1,study.get_psg_shape()[0]/study.sample_rate + 1)
    # y=np.reshape(y, [psg_shape[0], int(psg_shape[1] / study.sample_rate) , 1])
    study.y = np.reshape(_y, [psg_shape[0], int(psg_shape[1] / study.sample_rate) , 2])


print(model.input.shape)
print(model.layers[-1].output.shape)
print(dev_studies[0].get_all_periods().shape)
print(dev_studies[0].y.shape)
all_studies = np.concatenate([np.array(dev_studies),np.array(test_studies)])


# In[91]:


# df = pd.DataFrame({"Name": [s.psg_file_path.split("\\")[-1].replace(".edf","") for s in all_studies],
#                     "MS": [np.any(s.y[...,-1]==1) for s in all_studies],
#                     "neg": [np.sum(s.y[...,0]) for s in all_studies],
#                     "pos": [np.sum(s.y[...,1]) for s in all_studies],
#                     "n_periods": [x.n_periods for x in all_studies],
#                     "Studies": all_studies})
# weight_for_0 = (1 / sum(df.neg)) * (sum(df.neg+df.pos) / 2.0)
# weight_for_1 = (1 / sum(df.pos)) * (sum(df.pos + df.neg) / 2.0)

# class_weight = {0: weight_for_0, 1: weight_for_1}

# print('Weight for class 0: {:.2f}'.format(weight_for_0))
# print('Weight for class 1: {:.2f}'.format(weight_for_1))


# In[92]:


from sklearn.model_selection import train_test_split
import pandas as pd


df_train = pd.DataFrame({"Name": splits['train'],
                        "MS": [np.any(s.y[...,1]==1) for s in dev_studies],
                        "n_periods": [x.n_periods for x in dev_studies],
                        "Studies": dev_studies})

_train, _val = train_test_split(df_train, test_size=0.2, stratify=df_train.MS, random_state=42)
train_studies = _train[_train.n_periods >= 0].Studies.tolist()
val_studies = _val[_val.n_periods >= 0].Studies.tolist()

del [_train, _val, df_train, dev_studies]


# In[93]:


import tensorflow as tf
class Generator(tf.keras.utils.Sequence):

    def __init__(self, studies: typing.List[SleepStudy], hparams: YAMLHParams):
        
        self.studies = studies
        self.params  = hparams
        self.batch_shape = hparams['build']['batch_shape']
        self.batch_size = self.batch_shape[0]
        self.period_size = self.batch_shape[1]
        self.n_classes = 2
        self.n_channels = 4
        self.margin = int(np.floor((self.period_size / 2)))
        self.num_entries = sum([(x.get_all_periods().shape[0] - self.margin*2) * x.n_channels for x in studies])

        # Init x and y
        self.indices = self._generate_indices()
        self.idx = np.arange(0, len(self.indices))
        self.x = self._get_x_data()
        self.y = self._get_y_data()

        np.random.shuffle(self.idx)
   
    def __len__(self):
        return int(np.floor(self.num_entries / self.batch_size))

    def __getitem__(self, index):
        inds = self.idx[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y
    
    def _generate_indices(self):
        num_pi=[x.n_periods - self.margin*2 for x in self.studies]
        idxs = []
        for s in range(len(self.studies)):
            for p in range(int(num_pi[s])):
                for c in range(self.n_channels):
                    idxs.append((s, p+self.margin, c))
        return idxs

    def _get_x_data(self): 
        
        def _get_psg_by_idx(s: SleepStudy, period, channel, margin):
            return s.get_all_periods()[period-margin:period+margin+1,...,s.channel_groups[channel].channel_indices]

        return np.array([_get_psg_by_idx(self.studies[i[0]], i[1], i[2], self.margin) for i in self.indices])
    
    def _get_y_data(self): 
        
        def _get_target_by_idx(s: SleepStudy, period, margin):
            return s.y[period-margin:period+margin+1, ...]

        return np.array([_get_target_by_idx(self.studies[i[0]], i[1], self.margin) for i in self.indices])
    

    def on_epoch_end(self):
        np.random.shuffle(self.idx)

val_data = Generator(val_studies, hparams)
train_data = Generator(train_studies, hparams)  


# In[94]:


train_pos = sum([np.sum(v.y[...,1]==1) for v in train_studies])/sum([np.prod(v.y[...,1].shape) for v in train_studies])
train_neg = 1 - train_pos
val_pos = sum([np.sum(v.y[...,1]==1) for v in val_studies])/sum([np.prod(v.y[...,1].shape) for v in val_studies])
val_neg = 1 - val_pos

print(f"Training {train_pos:.4f} are positive - Naive method baseline is: {train_neg:.4f}")
print(f"Validation {val_pos:.4f} are positive - Naive method baseline is: {val_neg:.4f}")


# In[95]:


import numpy as np 
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


## Train

# In[97]:

from keras.callbacks import EarlyStopping

epochs_pre = 50
cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = tf.keras.metrics.CategoricalAccuracy())

with tf.device("/device:GPU:0"):
    history = model.fit(train_data,
                        validation_data=val_data, 
                        epochs=epochs_pre,
                        callbacks=cb)

import pickle
with open('history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# In[ ]:


model.save_weights("weights_100_pre.h5")


# In[102]:

epochs_post = epochs_pre + 50

# Unfreeze for fine-tuning
base.trainable = True

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = tf.keras.metrics.CategoricalAccuracy())

cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

with tf.device("/device:GPU:0"):
    history_fine = model.fit(train_data,
                            validation_data=val_data, 
                            epochs=epochs_post,
                            initial_epoch=history.epoch[-1],
                            callbacks=cb)


# In[ ]:


model.save_weights("weights_100_post.h5")

with open('history_fine', 'wb') as file_pi:
    pickle.dump(history_fine.history, file_pi)


# In[ ]:


# from sklearn.metrics import confusion_matrix

# y_hat = []
# y_prob = []
# y_true = []
# for study in val_studies:
#     base = get_and_load_one_shot_model(
#                 study.n_periods,
#                 project_dir=args.project_dir,
#                 hparams=hparams,
#             )

#     # Extract base from pre-trained model (remove last )
#     extract = base.layers[-5].output

#     # Create new head with base model as input with a 2-class problem
#     head=base.create_seq_modeling(in_=extract,
#                                 input_dims=base.input_dims,
#                                 data_per_period=args.data_per_prediction,
#                                 n_periods=base.n_periods,
#                                 n_classes=2,
#                                 transition_window=base.transition_window,
#                                 activation=base.activation,
#                                 regularizer=None)


#     pred_model = Model(inputs=base.input, outputs = head, name = "mU-Sleep")
#     pred_model.load_weights("weights_19_long.h5")

#     print(f"Input shape: {pred_model.input.shape}")
#     print(f"Output shape: {pred_model.output.shape}")
#     clear_output(wait=False)

#     prob = np.empty([len(study.channel_groups), study.n_periods*pred_model.output_shape[-2], pred_model.output_shape[-1]])
#     for i, channel_group in enumerate(study.channel_groups):
#         # Get PSG for particular group
#         psg = np.expand_dims(study.get_all_periods(),0)
#         psg_subset = psg[..., tuple(channel_group.channel_indices)]
#         prob_i = pred_model.predict_on_batch(psg_subset)
#         prob[i,...] = prob_i.reshape(-1, prob.shape[-1])

#     mean_prob = prob.mean(axis=0)
#     pred = np.argmax(mean_prob,axis=1)
#     target = study.y[...,1].flatten()

#     print(confusion_matrix(target, pred))
    
#     y_prob.append(mean_prob)
#     y_hat.append(pred)
#     y_true.append(target)
# yhat=np.concatenate(y_prob)[:,1]
# testy = np.concatenate(y_true)


# In[ ]:


# # pr curve for logistic regression model
# from sklearn.metrics import precision_recall_curve
# from matplotlib import pyplot

# # calculate pr-curve
# precision, recall, thresholds = precision_recall_curve(testy, yhat)
# # plot the roc curve for the model
# no_skill = len(testy[testy==1]) / len(testy)
# pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
# pyplot.plot(recall, precision, marker='.', label='Logistic')
# # axis labels
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# pyplot.legend()
# # show the plot
# pyplot.show()

# # convert to f score
# fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
# ix = np.argmax(fscore)
# print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))


# In[ ]:


# from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
# from pprint import pprint


# y_hat = []
# y_prob = []
# y_true = []
# for study in test_studies:
#     base = get_and_load_one_shot_model(
#                 study.n_periods,
#                 project_dir=args.project_dir,
#                 hparams=hparams,
#             )

#     # Extract base from pre-trained model (remove last )
#     extract = base.layers[-5].output

#     # Create new head with base model as input with a 2-class problem
#     head=base.create_seq_modeling(in_=extract,
#                                 input_dims=base.input_dims,
#                                 data_per_period=args.data_per_prediction,
#                                 n_periods=base.n_periods,
#                                 n_classes=2,
#                                 transition_window=base.transition_window,
#                                 activation=base.activation,
#                                 regularizer=None)


#     pred_model = Model(inputs=base.input, outputs = head, name = "mU-Sleep")
#     pred_model.load_weights("weights_19_long.h5")

#     print(f"Input shape: {pred_model.input.shape}")
#     print(f"Output shape: {pred_model.output.shape}")
#     clear_output(wait=False)

#     prob = np.empty([len(study.channel_groups), study.n_periods*pred_model.output_shape[-2], pred_model.output_shape[-1]])
#     for i, channel_group in enumerate(study.channel_groups):
#         # Get PSG for particular group
#         psg = np.expand_dims(study.get_all_periods(),0)
#         psg_subset = psg[..., tuple(channel_group.channel_indices)]
#         prob_i = pred_model.predict_on_batch(psg_subset)
#         prob[i,...] = prob_i.reshape(-1, prob.shape[-1])

#     mean_prob = prob.mean(axis=0)
#     pred = np.argmax(mean_prob,axis=1)
#     target = study.y[...,1].flatten()

#     print(confusion_matrix(target, pred))
    
#     y_prob.append(mean_prob)
#     y_hat.append(pred)
#     y_true.append(target)



# In[ ]:


# from sklearn.metrics import f1_score, recall_score, precision_score, cohen_kappa_score
# y_pred = np.concatenate(y_prob)[:,1] > thresholds[ix]
# y_target = np.concatenate(y_true)
# print(f"Cohen kappa: {cohen_kappa_score(y_target, y_pred):.3f}")
# print(confusion_matrix(y_target, y_pred))
# print(f1_score(y_target, y_pred))
# print(precision_score(y_target, y_pred))
# print(recall_score(y_target, y_pred))


# In[ ]:


# ## Initialize tf.Sequence object with list of SleepStudies with the following ´__get_item()__´ functionality
# #     1. Select random sleep study (or balanced?)
# #     2. Select random period with margin (limit the randomness within margins)
# #     3. Select random channels
# #     4. Extract psg based on the above
# #     5. Extract y (labels) with the same indexing as above
# #     6. Repeat for each batch
# import tensorflow as tf
# from tensorflow.keras.utils import Sequence

# class MyBaseSequence(Sequence):

#     def __init__(self, studies: typing.List[SleepStudy], hparams: YAMLHParams):
        
#         self.studies = studies
#         self.params  = hparams
#         self.batch_shape = hparams['build']['batch_shape']
#         self.batch_size = self.batch_shape[0]
#         self.period_size = self.batch_shape[1]
#         self.n_classes = 2
#         self.margin = np.floor((self.period_size / 2))
#         self.period_length_sec = hparams['prediction_params']['period_length']
#         self.num_entries = sum([(x.get_all_periods().shape[0] - self.margin*2) * x.n_channels for x in studies])
#         self.num_sequences = np.ceil(self.num_entries / self.batch_size)
        

#     def __len__(self):
#         return int(self.num_sequences)

#     # def get_study(self):
#     #     return

#     # def get_period_idx(self):
#     #     return
        
#     def _process_batch(self):

#         # Get study
#         _study = self.get_study()

#         # Get period
#         period_idx = self.get_period_idx(_study)

#         # Get channels
#         channels = self.get_channels(_study)
        
#         # Extract PSG from above
#         psg = _study.get_all_periods()
#         psg_subset = psg[int(period_idx-self.margin):int(period_idx+self.margin)+1,...,channels]

#         # Extract targets from above
#         target_subset = _study.y[int(period_idx-self.margin):int(period_idx+self.margin)+1,...]

#         return psg_subset, target_subset

#     def get_batches(self):

#         X = np.empty(shape=self.batch_shape)
#         Y = np.empty(shape=[self.batch_size, self.period_size, self.period_length_sec, self.n_classes])

#         for i in range(self.batch_size):
#             X[i,...], Y[i,...] = self._process_batch() 
        
#         return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)

#     def __getitem__(self, idx):
#         return self.get_batches()


# class MyRandomBatchSequence(MyBaseSequence):

#     def __init__(self, studies: YAMLHParams, hparams: YAMLHParams):
#         super().__init__(studies, hparams)

#     def get_study(self):
#         return np.random.choice(self.studies, 1)[0]
    
#     def get_period_idx(self, study):
#         upper_margin_limit = int(study.n_periods - self.margin)
#         return np.random.randint(self.margin, upper_margin_limit, 1)

#     def get_channels(self, study):
#         # Select random channels
#         ch_idx = np.random.randint(0, len(study.channel_groups))
#         return study.channel_groups[ch_idx][1]


# class MyBalancedBatchSequence(MyBaseSequence):

#     def __init__(self, studies: YAMLHParams, hparams: YAMLHParams):
#         super().__init__(studies, hparams)
#         self._study_idx = 0
#         self._period_idx = self.margin
#         self._ch_idx = 0

#     def get_study(self):
#         return self.studies[self._study_idx]
    
#     def get_period_idx(self, study = None):
#         return self._period_idx

#     def get_channels(self, study):
#         return study.channel_groups[self._ch_idx][1]
    
#     def _shift(self):
        
#         self._ch_idx += 1
#         if self._ch_idx == 4:
#             self._ch_idx = 0
#             self._period_idx += 1
        
#         if self._period_idx == self.studies[self._study_idx].n_periods - self.margin:
#             self._study_idx += 1
#             self._period_idx = self.margin
#             self._ch_idx = 0

#         if self._study_idx == len(self.studies):
#             self._study_idx = 0
#             self._period_idx = self.margin
#             self._ch_idx = 0

#         return

#     def get_batches(self):

#         X = np.empty(shape=self.batch_shape)
#         Y = np.empty(shape=[self.batch_size, self.period_size, self.period_length_sec, self.n_classes])

#         for i in range(self.batch_size):
#             X[i,...], Y[i,...] = self._process_batch()
#             self._shift()
        
#         return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)




# random_train_dataset = MyRandomBatchSequence(train_studies, hparams)
# random_val_dataset = MyRandomBatchSequence(val_studies, hparams)


# balanced_train_dataset = MyBalancedBatchSequence(train_studies, hparams)
# balanced_val_dataset = MyBalancedBatchSequence(val_studies, hparams)

