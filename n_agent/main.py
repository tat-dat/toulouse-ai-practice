#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:33:01 2018

@author: tthnguye
"""
import fmodel
import keras.backend as K
from fmodel import Lmodel,Vmodel,mumodel
from env import Env
import numpy as np
import gym
from keras.optimizers import Adam
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.
N=5
env=Env(N)
nb_actions =N

processor = PendulumProcessor()

memory = SequentialMemory(limit=100000, window_length=1)

random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, 
                                          size=nb_actions)

agent = NAFAgent(covariance_mode='diag',nb_actions=nb_actions, V_model=Vmodel(N), L_model=Lmodel(N),
                 mu_model=mumodel(N),memory=memory, nb_steps_warmup=100, 
                 random_process=random_process, gamma=.99, target_model_update=1e-3)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=150, visualize=False, verbose=0, nb_max_episode_steps=20)

agent.save_weights('weight1.dhf5', overwrite=True)