#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:49:15 2018

@author: tthnguye
"""
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.layers import Input,core, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

#to slice tensor x, x[1,a:b]: i.e. slice at place 1, from a to b. if the first place is
#: then slice all place from a to b.
def Lmodel(N):
    a = Input(batch_shape=(1,N))
    ob= Input(batch_shape=(1,N))
    ain=K.transpose(a)
    obin=K.transpose(ob)
    x=concatenate([ain,obin])
    x=Dense(12)(x)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x=K.transpose(x)
    model=Model(inputs=[a,ob],outputs=x)
    return model
#    print(model.summary())

def Vmodel(N):
    inpp=Input(batch_shape=(1,N))
    inp=K.transpose(inpp)
    x=Dense(12)(inp)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    out=K.transpose(x)
    model=Model(inputs=inpp,outputs=out)
    return model

def mumodel(N):
    inpp=Input(batch_shape=(1,N))
    inp=K.transpose(inpp)
    x=Dense(12)(inp)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    out=K.transpose(x)
    model=Model(inputs=inpp,outputs=out)
    return model