# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 01:25:38 2022

@author: Maysam
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.layers import GINConv, GlobalAvgPool
from ARMA_Func import computeLoaderCombine2,computeLoaderCombine5,TimeHistory,get_phsi

from ARMA_Net import ARMAConv
from tensorflow.keras.regularizers import l2
################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
channels = 128  # Hidden units
layers = 2  # ARMA layers 3
epochs = 200  # Number of training epochs
batch_size = 32  # Batch size  32 10

################################################################################
# LOAD DATA
# ENZYMES  PROTEINS MUTAG
################################################################################
dataset = TUDataset("ENZYMES", clean=True)
dataset=dataset[100:200]
# idxs = np.random.permutation(100)
# dataset=dataset[idxs]

# Parameters
F = dataset.n_node_features  # Dimension of node features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.8 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

# loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
# loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)
steps_per_epoch_tr=int(np.ceil(len(dataset_tr) / batch_size))
steps_per_epoch_te=int(np.ceil(len(dataset_te) / batch_size))
################################################################################
# Compute data loader 
################################################################################

data_tr=dataset[idx_tr]
data_te=dataset[idx_te]

# loader_tr,loader_te_load=computeLoaderCombine2(data_tr,
#       loader_tr,loader_te,apx_phsi,N_scales,scales,m,epochs,thr)

loader_tr,_=computeLoaderCombine5(data_tr,epochs)

loader_te,_=computeLoaderCombine5(data_te,epochs)
# psi,psii=get_phsi(data_tr,apx_phsi,N_scales,scales,m,epochs,thr)
# psi_te,psi_inv_te=get_phsi(loader_te,apx_phsi,N_scales,scales,m,epochs,thr)
################################################################################
# Parmeters
# channels = 16  # Number of features in the first layer
iterations =1  # Number of layers
share_weights = False  # Share weights 
dropout_skip = 0.75  # Dropout rate for the internal skip connection 
l2_reg = 5e-4  # L2 regularization rate


# BUILD MODEL
################################################################################

class ARMA(Model):
    def __init__(self, channels, n_layers):
        super().__init__()
        self.conv1=ARMAConv(channels,iterations=iterations,order=1,
                           share_weights=share_weights,dropout_rate=dropout_skip,
                           activation="elu",gcn_activation="elu",kernel_regularizer=l2(l2_reg),)
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                ARMAConv(channels,iterations=iterations,order=1,
                        share_weights=share_weights,dropout_rate=dropout_skip,
                        activation="elu",gcn_activation="elu",kernel_regularizer=l2(l2_reg),)            )
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_out, activation="softmax")

    def call(self, inputs):
        x, L, a, i= inputs
        x = self.conv1([x, L, a])
        for conv in self.convs:
            x = conv([x, L, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


# Build model
model = ARMA(channels, layers)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()


################################################################################
# FIT MODEL
################################################################################
# @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs,target):
    with tf.GradientTape() as tape:
        # inputs=list(inputs)
        # inputs.append(psi_tr)
        # inputs.append(psii_tr)
        # inputs=tuple(inputs)
        predictions = model(inputs,training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc

b=[]
print("Fitting model")
current_batch = 0
model_lss = model_acc = 0
for batch in loader_tr:
    # batch0=batch[0]
    # batch0=list(batch0)
    # batch0.append(psi)
    # batch0=tuple(batch0)
    # batch=list(batch)
    # batch[0]=batch0
    # batch=tuple(batch)
    lss, acc = train_step(*batch)

    model_lss += lss.numpy()
    model_acc += acc.numpy()
    current_batch += 1
    # if current_batch == loader_tr.steps_per_epoch:
        # model_lss /= loader_tr.steps_per_epoch
        # model_acc /= loader_tr.steps_per_epoch
    if current_batch == steps_per_epoch_tr:
        model_lss /= steps_per_epoch_tr
        model_acc /= steps_per_epoch_tr
        print("Loss: {}. Acc: {}".format(model_lss, model_acc))
        model_lss = model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model_lss = model_acc = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    model_lss += loss_fn(target, predictions)
    model_acc += tf.reduce_mean(categorical_accuracy(target, predictions))
# model_lss /= loader_te.steps_per_epoch
# model_acc /= loader_te.steps_per_epoch
model_lss /= epochs
model_acc /= epochs
print("Done. Test loss: {}. Test acc: {}".format(model_lss, model_acc))


# get parameters 
ww=model.get_weights()