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

    def __init__(self,scenario,name=""):
        super().__init__(scenario,name=name)

        envSpec=self.environmentSpec
        self.kpiNum=2
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum+self.actionNum+self.kpiNum,), dtype=np.float32, minimum=np.append(envSpec.minState,[0]*(self.actionNum+self.kpiNum)), maximum=np.append(envSpec.maxState,[1]*(self.actionNum+self.kpiNum)),name='observation')
        self._observation_spec_no_mask = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum,), dtype=np.float32, minimum=envSpec.minState, maximum=envSpec.maxState,name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.actionNum-1,name='action')
     
    
    def observation_spec(self) :
        return self._observation_spec

    def action_spec(self) :
        return self._action_spec  
    @property
    def batched(self) -> bool:
        return False
    @property
    def batch_size(self):
        return 0
    
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
      
    def _reset(self): 
        self.start()  
        return ts.restart(self.getObservation())
    
    def _step(self,actionIdx): 
        if self.finishedEpisode():
            return self._reset()
        
        super().takeAction(actionIdx) 
        observ=self.getObservation() 
        if self.finishedEpisode() : 
            return ts.termination(observ, reward=self.reward)
         
        return ts.transition(observ, reward=self.reward, discount=0.5) 
    
