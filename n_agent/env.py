#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:57:45 2018

@author: tthnguye
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
#    reward_range = (-np.inf, np.inf)
#    action_space = None
#    observation_space = None
    def __init__(self,N):
        self.Nn=N
#        self.range = 1000  # +/- value the randomly select number can be between
#        self.bounds = 2000  # Action space bounds

#        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.bounds]))
#        self.observation_space = spaces.Discrete(4)
        self.ac=[]
        self.rw=[]
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        Cr=2
        Cs=0.4
        Cd=2
        D=[]
        mu=0.5
        lam=0.3
        R=2*(Cd+Cr+Cs/mu)
        reward=[]
        N=self.Nn
        meet=np.random.exponential(1/lam,size=N)
        self.ac.append(action)
        for i in range(N):
            assert action[i]!=np.nan
            if meet[i]<action[i]:
                D.append(meet[i]+np.random.exponential(1/mu))
                reward.append(-Cr-Cs*(D[-1]-meet[i]))
            else:
                D.append(np.inf)
                reward.append(0)
        ind=D.index(min(D))
        pro=int(D[ind] is not np.inf)
        if pro==1:
            reward[ind]=reward[ind]+R-Cd
        ob=[]
        for i in range(N):
            if reward[i]>0:# be the first who meets the Destination and has positive reward
                ob.append([action[i],4])
            elif (i==ind and D[ind] is not np.inf):#be the first who meets D but has negative reward
                ob.append([action[i], 3])
            elif meet[i]<action[i]:
                ob.append([action[i],2])
            else:
                ob.append([action[i],1])
        self.rw.append(reward)
  #      reward=[np.mean(np.array(self.rw)[:,i]) for i in range(self.Nn)]
        return ob,reward,False,{}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        N=self.Nn
        Cr = 2
        Cs = 0.4
        Cd = 2
        mu = 0.5
        lam = 0.3
        R = 2 * (Cd + Cr + Cs / mu)
        reward=[]
        D=[]
        meet = np.random.exponential(1 / lam, size=N)
        action=[random.randint(0,10) for _ in range(N)]
        for i in range(N):
            assert action[i]!=np.nan
            if meet[i]<action[i]:
                D.append(meet[i]+np.random.exponential(1/mu))
                reward.append(-Cr-Cs*(D[-1]-meet[i]))
            else:
                D.append(np.inf)
                reward.append(0)
        ind=D.index(min(D))
        pro=int(D[ind] is not np.inf)
        if pro==1:
            reward[ind]=reward[ind]+R-Cd
        ob=[]
        for i in range(N):
            if reward[i]>0:# be the first who meets the Destination and has positive reward
                ob.append([action[i],4])
            elif (i==ind and D[ind] is not np.inf):#be the first who meets D but has negative reward
                ob.append([action[i], 3])
            elif meet[i]<action[i]:
                ob.append([action[i],2])
            else:
                ob.append([action[i],1])
        return ob
