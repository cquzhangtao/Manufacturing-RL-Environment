'''
Created on Dec 4, 2021

@author: Shufang
'''

import numpy as np

from com.tao.py.rl.environment.Environment4 import SimEnvironment4


import gym
from gym import spaces



class SimEnvironment7(SimEnvironment4,gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,scenario,name=""):
        super().__init__(scenario,name=name)
        
        envSpec=self.environmentSpec
        self.action_space = spaces.Discrete(self.actionNum)
        
        self.observation_space = spaces.Space(shape=(envSpec.stateFeatureNum,), dtype=np.float32)
        
        #
        # self.kpiNum=2
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(envSpec.stateFeatureNum+self.actionNum+self.kpiNum,), dtype=np.float32, minimum=np.append(envSpec.minState,[0]*(self.actionNum+self.kpiNum)), maximum=np.append(envSpec.maxState,[1]*(self.actionNum+self.kpiNum)),name='observation')
        # self._observation_spec_no_mask = array_spec.BoundedArraySpec(
        #     shape=(envSpec.stateFeatureNum,), dtype=np.float32, minimum=envSpec.minState, maximum=envSpec.maxState,name='observation')
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=0, maximum=self.actionNum-1,name='action')
     
    def step(self, actionIdx):
        super().takeAction(actionIdx) 
        observ=self.getObservation() 
        return observ, self.reward, self.finishedEpisode(), None
    
    def reset(self):
        self.start()  
        observ=self.getObservation() 
        return observ  # reward, done, info can't be included
    def render(self, mode='human'):
        pass
    def close (self):
        pass
    
    def restart(self):
        pass
    
    def getObservation(self):
        kpi=[]
        if self.finishedEpisode():
            if len(self.kpi)<1:
                kpi=[0,0]
            kpi=[self.kpi[len(self.kpi)-1],self.rewards[len(self.rewards)-1]]
        else:
            if len(self.kpi)<1:
                kpi=[0,0]
            else:
                kpi=[self.kpi[len(self.kpi)-1],self.rewards[len(self.rewards)-1]]
            
            
        
        return np.array(self.state.getData()+self.getMask()+kpi, dtype=np.float32)      