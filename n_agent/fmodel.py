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
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

#to slice tensor x, x[1,a:b]: i.e. slice at place 1, from a to b. if the first place is
#: then slice all place from a to b.
def Lmodel(N):
    a = Input(shape=(N,))
    ob= Input(shape=(1,N))
    transpose_layers = Lambda(lambda v: K.transpose(v))
    ain=transpose_layers(a)
    obin=transpose_layers(ob)
    x=concatenate([ain,obin])
    x=Dense(12)(x)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x=Dense((N*N+N)//2)(x)
    x=transpose_layers(x)
    model=Model(inputs=[a,ob],outputs=x)
    return model
#    print(model.summary())

def Vmodel(N):
    inpp=Input(shape=(1,N))
    transpose_layers = Lambda(lambda v: K.transpose(v))
    inp=transpose_layers(inpp)
    x=Dense(12)(inp)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x=Dense(N)(x)
    out=transpose_layers(x)
    model=Model(inputs=inpp,outputs=out)
    return model

def mumodel(N):
    inpp=Input(shape=(1,N))
    transpose_layers = Lambda(lambda v: K.transpose(v))
    inp=transpose_layers(inpp)
    x=Dense(12)(inp)
    x=Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x=Dense(N)(x)
    out=transpose_layers(x)
    model=Model(inputs=inpp,outputs=out)
    return model