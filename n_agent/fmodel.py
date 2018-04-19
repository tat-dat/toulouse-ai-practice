#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:49:15 2018

@author: tthnguye
"""
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.layers import Lambda, Input,core, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.core import Reshape
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

#to slice tensor x, x[1,a:b]: i.e. slice at place 1, from a to b. if the first place is
#: then slice all place from a to b.
def Lmodel(N):
    a = Input(shape=(N,))
    ob= Input(shape=(N,2))
    expand_dims = Lambda(lambda v: K.expand_dims(v))
 #   permutation_layers=Lambda(lambda v: K.permute_dimensions(v,(0,2,1)))
#    ain=transpose_layers(a)
  #  obin=permutation_layers(ob) #become: (?,5,2)
    ain = expand_dims(a)    # become (?,5,1)
    x=concatenate([ain,ob])
    x=Dense(40)(x,activation='relu')
    x = Dense(40)(x,activation='relu')
    x = Dense(40)(x,activation='relu')
    x=Dense(1)(x,activation='sigmoid')
    x = Flatten()(x)
    model=Model(inputs=[a,ob],outputs=x)  #output is (?,5)
    return model
#    print(model.summary())

def Vmodel(N):
    inpp=Input(shape=(N,2))
#    transpose_layers = Lambda(lambda v: K.transpose(v))
 #   permutation_layers=Lambda(lambda v: K.permute_dimensions(v,(0,2,1)))
 #   inp=permutation_layers(inpp)
#    expand_dims = Lambda(lambda v: K.expand_dims(v,axis=0))
    x=Dense(16)(inpp,activation='relu')
    x = Dense(16)(x,activation='relu')
    x = Dense(16)(x,activation='relu')
    x=Dense(1)(x,activation='linear')
    x = Flatten()(x)
    model=Model(inputs=inpp,outputs=x) # out is (?, 5)
    return model

def mumodel(N):
    inpp=Input(shape=(N,2))
   # permutation_layers=Lambda(lambda v: K.permute_dimensions(v,(0,2,1)))
   # inp=permutation_layers(inpp)
    x=Dense(16)(inpp,activation='relu')
    x = Dense(16)(x,activation='relu')
    x = Dense(16)(x,activation='relu')
    x=Dense(1)(x,activation='relu')
    x = Flatten()(x)
    model=Model(inputs=inpp,outputs=x) # out is (?, 5)
    return model