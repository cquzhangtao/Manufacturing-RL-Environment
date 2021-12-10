'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import array_spec


from com.tao.py.rl.environment.Environment4 import SimEnvironment4





class SimEnvironment5(SimEnvironment4,PyEnvironment):

    def __init__(self,scenario):
        super().__init__(scenario)

        envSpec=self.environmentSpec
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum,), dtype=np.float32, minimum=envSpec.minState, maximum=envSpec.maxState,name='observation')
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.actionNum,name='action')
     
    
    def observation_spec(self) :
        return self._observation_spec

    def action_spec(self) :
        return self._action_spec  
    
     
    def _reset(self): 
        self.start()  
        return ts.restart(np.array(self.state.getData(), dtype=np.float32))
    
    def _step(self,actionIdx): 
        super().takeAction(actionIdx)   
        return ts.transition(
          np.array(self.state.getData(), dtype=np.float32), reward=self.reward, discount=1.0) 
    
