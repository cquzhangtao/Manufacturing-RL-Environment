'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment

from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np


class CardGameEnv(PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=[1,2],maximum=[5,6], name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=(1,), dtype=np.int32, name='observation')
        self._state = 0
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
    
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        

        new_card = np.random.randint(1, 11)
        self._state += new_card

        
        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
              np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
            
            
            
            