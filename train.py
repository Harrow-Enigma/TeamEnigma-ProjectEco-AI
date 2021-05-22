"""
Project ECO AI Model

---------------------------------------------------------

Copyright 2021 YIDING SONG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import requests
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.titlecolor'] = 'green'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'

# Input dimensions
INP_DIM = 3

# The number of past time steps the model will be given data from
HIST_LEN = 10

# Indices of the features to be fed into the model at each time step
FEAT = [i for i in range(INP_DIM)]

# The number of time steps the model will need to predict into the future
PRED_LEN = 3

# Indices of the labels to be predicted by the model at each time step
LAB = [i for i in range(INP_DIM)]

# Whether your RNN model will return sequences
RET_SQN = True

BATCH_SZ = 64

RNN_UNITS = 256

assert set(LAB).issubset(set(FEAT))

mapping = {i:e for e,i in enumerate(FEAT)}

# Indices of the labels to be predicred, relative to the input array
REL_LAB = [mapping[i] for i in LAB]

"""Loading the data"""

# prepares `raw_data` as a np.float32 array where
# x-axis is a list of data points from the same time
# y-axis is follows the flow of time

def dict2date(data_dict):
  return dt.strptime(data_dict['DATE'], '%Y-%m-%dT%H:%M:%S.%f%z')

api_url = 'https://projecteco-webserver.dylankainth.repl.co/api/rest/ai/v1'
response = requests.get(api_url)
json_data = response.json()
json_data = sorted(json_data, key = dict2date)

raw_data = []

for t in json_data:
  temp = []
  for loc in [l for l in t.keys() if l not in ['__v', '_id', 'DATE']]:
    temp.append([t[loc][s] for s in t[loc].keys()])
  raw_data.append(np.array(temp).mean(0))

raw_data = np.array(raw_data, np.float32)

print(raw_data.shape)

"""Data Preprocessing"""

def standardize(arr):
  m = arr.mean(0)
  s = arr.std(0)
    
  for i in range(len(s)):
    if s[i] == 0:
      s[i] = 1e-8
  
  arr = (arr - m)/s
  return arr, m, s

data, mean, std = standardize(raw_data)

print('Array of standardized data (first 2)')
print(data[:2])
assert raw_data.shape == data.shape
assert (data[0] * std + mean).all() == raw_data[0].all()
assert data[1][2] * std[2] + mean[2] == raw_data[1][2]

def window(data, hist_len, feat, pred_len, lab, ret_sqn):
  seg_len = hist_len + pred_len
  feat_len = hist_len
  lab_len = 1
  if ret_sqn:
    lab_len += feat_len-1
  feat_dim = len(feat)
  lab_dim = len(lab) * pred_len

  features_dataset = []
  labels_dataset = []

  for i in range(len(data) - seg_len + 1):
    seg = data[i:i+seg_len]
  
    features = []
    for f in seg[:feat_len]:
      features.append([f[j] for j in feat])
    features_dataset.append(features)

    labels = []
    for f in range(seg_len-pred_len-lab_len+1, seg_len-pred_len+1):
      local_lab = []
      for j in range(f, f+pred_len):
        local_lab.extend(seg[j][k] for k in lab)
      labels.append(local_lab)
    labels_dataset.append(labels)
  
  assert len(features_dataset) == len(labels_dataset)

  dataset = tf.data.Dataset.from_tensor_slices((features_dataset, labels_dataset))
  
  return dataset, len(features_dataset), (feat_len, feat_dim), (lab_len, lab_dim)

dataset, dataset_size, feature_shape, label_shape = window(
    data, HIST_LEN, FEAT, PRED_LEN, LAB, RET_SQN
)
print('Dataset {} of size {}'.format(dataset, dataset_size))
print('Shape of features: {}'.format(feature_shape))
print('Shape of labels: {}'.format(label_shape))

for i in dataset.take(1):
  sample_features = i[0]
  sample_labels = i[1]
  print('Sample features:\n{}\n\nSample labels:\n{}\n\n'.format(
      sample_features, sample_labels
  ))
  assert i[0].shape == feature_shape and i[1].shape == label_shape

ratio = 9/10
train_no = int(ratio * dataset_size)
test_no = dataset_size - train_no
print('Number of training samples: {}'.format(train_no))
print('Number of testing samples: {}'.format(test_no))

train_dataset = dataset.take(train_no)
test_dataset = dataset.skip(train_no)
train_dataset = train_dataset.batch(BATCH_SZ)
test_dataset = test_dataset.batch(BATCH_SZ)

json.dump({
    'mean': mean.tolist(),
    'std': std.tolist(),
    'hist_len': HIST_LEN,
    'features': FEAT,
    'pred_len': PRED_LEN,
    'labels': LAB
}, open('data_aux.json', 'w'))

"""Pre-Training Data"""

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

df.head()

for e, i in enumerate(list(df.columns)):
    print(e, '\t', i)

pre_raw = df.iloc[:, [2, 5, 10]].to_numpy()
pre_data, pre_mean, pre_std = standardize(pre_raw)
print(pre_data.shape)

assert pre_data.shape[1] == INP_DIM

pre_dataset, pre_dataset_size, _, _ = window(
    pre_data, HIST_LEN, FEAT, PRED_LEN, LAB, RET_SQN
)

pre_ratio = 9.5/10
pre_train_no = int(pre_ratio * pre_dataset_size)
pre_test_no = pre_dataset_size - pre_train_no

pre_train_dataset = pre_dataset.take(pre_train_no)
pre_test_dataset = pre_dataset.skip(pre_train_no)
pre_train_dataset = pre_train_dataset.batch(BATCH_SZ)
pre_test_dataset = pre_test_dataset.batch(BATCH_SZ)

"""Building the model"""

class ResidualForecastModel(tf.keras.Model):
  def __init__(self, input_shape, rel_lab, pred_len,
               ret_sqn, rnn_units, name = 'ResidualForecastModel'):
    super(ResidualForecastModel, self).__init__(name = name)
    self.inp_shape = input_shape
    self.rel_lab = rel_lab
    self.pred_len = pred_len
    self.ret_sqn = ret_sqn

    self.lab_len = len(self.rel_lab)
    self.sqn_len = self.inp_shape[0]
    self.out_dim = self.lab_len * self.pred_len
    
    self.rnn = tf.keras.layers.GRU(
      rnn_units, input_shape = input_shape,
      return_sequences = ret_sqn
    )
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.dense = tf.keras.layers.Dense(128, activation = 'tanh')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.out = tf.keras.layers.Dense(self.out_dim)
  
  def cumulative_sum(self, inp, pred):
    base_sum = tf.gather(inp, self.rel_lab, axis = -1)
    batch_size = tf.shape(pred)[0]
    sum_matrix = tf.reshape(base_sum, [
        batch_size, self.sqn_len, 1, self.lab_len
    ])
    batchSz = pred.shape[0]
    pred = tf.reshape(pred, [
      batch_size, self.sqn_len, self.pred_len, self.lab_len
    ])

    for i in range(self.pred_len - 1):
      base_sum = base_sum + pred[:, :, i, :]
      sum_matrix = tf.concat([sum_matrix, tf.reshape(base_sum, [
          batch_size, self.sqn_len, 1, self.lab_len
      ])], -2)
    
    pred = pred + sum_matrix
    return tf.reshape(pred, (
        batch_size, self.sqn_len, self.out_dim
    ))
  
  def call(self, inp):
    x = self.bn1(self.rnn(inp))
    x = self.bn2(self.dense(x))
    x = self.out(x)
    return self.cumulative_sum(inp, x)
  
  def predict(self, inp):
    pred = self.call(inp)
    batch_size = tf.shape(pred)[0]
    return tf.reshape(pred[:, -1, :],
                      [batch_size, self.pred_len, self.lab_len])
  
  def functional(self):
    inputs = tf.keras.Input(self.inp_shape)
    outputs = self.call(inputs)
    return tf.keras.Model(inputs, outputs, name=self.name)

sample_model = ResidualForecastModel(
    feature_shape, REL_LAB, PRED_LEN, RET_SQN, RNN_UNITS
)

for i in pre_dataset.take(1):
  sample_pred = sample_model(tf.expand_dims(i[0], 0))
  print('Sample prediction of shape {}:\n{}'.format(
      sample_pred.shape, sample_pred
  ))

"""Model Visualization"""

sample_model.summary()

tf.keras.utils.plot_model(sample_model.functional(), to_file="model.png")

"""Defining losses and optimizers"""

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

optim = tf.keras.optimizers.Adam(1e-5)

model = ResidualForecastModel(
    feature_shape, REL_LAB, PRED_LEN, RET_SQN, RNN_UNITS
)
model.compile(optimizer = optim, loss = mse)

"""Defining training checkpoints"""

pre_checkpoint_dir = './ProjectECO_PreTraining_Checkpoints/'

if not os.path.exists(pre_checkpoint_dir):
  os.mkdir(pre_checkpoint_dir)

pre_checkpoint_prefix = os.path.join(pre_checkpoint_dir, "ckpt_{epoch}")

pre_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = pre_checkpoint_prefix,
    save_weights_only = True)

checkpoint_dir = './ProjectECO_Checkpoints/'

if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True)

"""TRAINING!!"""

## Pretraining
model.fit(pre_train_dataset,
          epochs = 4,
          callbacks=[pre_checkpoint_callback]
         )

model.evaluate(pre_test_dataset)

# Actual training
model.fit(train_dataset,
          epochs = 50,
          callbacks=[checkpoint_callback]
         )

model.evaluate(test_dataset)

json.dump({
  'input_shape': feature_shape,
  'rel_lab': REL_LAB,
  'pred_len': PRED_LEN,
  'ret_sqn': RET_SQN,
  'rnn_units': RNN_UNITS
}, open('model_aux.json', 'w'))

model.save_weights('weights.h5')
model.functional().save('model_func.h5')
