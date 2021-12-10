'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import array_spec


from com.tao.py.rl.environment.Environment3 import SimEnvironment3
from com.tao.py.rl.kernel.Action import Action
from com.tao.py.rl.kernel.State import State



class SimEnvironment5(SimEnvironment3,PyEnvironment):

    def __init__(self,scenario):
        super().__init__(scenario)
        self.init(100)
        envSpec=self.environmentSpec
        self.actionNum=0
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum,), dtype=np.float32, minimum=envSpec.minState, maximum=envSpec.maxState,name='observation')
        
                
        
        counter=envSpec.countAction
        
          
        print(counter)   
    
    
    def observation_spec(self) :
        return self._observation_spec

    def action_spec(self) :
        return None  
    
    def getActionIndex(self,action):
        pass     
     
    def _reset(self): 
        self.start()  
        return ts.restart(np.array([self.state.getData()], dtype=np.float32))
    
    def _step(self,actionIdx): 
        super().takeAction(actionIdx)   
        return ts.transition(
          np.array([self.state.getData()], dtype=np.float32), reward=self.reward, discount=1.0) 
    
    
    def getActionFromJob(self,job,time): 
        action= super().getActionFromJob(job, time)
        idx=self.getActionIndex(action)
        return Action([idx])
    
       
    def getActionSetFromQueue(self,queue,time):  
        actions=[]
        for job in queue:
            actions.append(self.getActionFromJob(job,time))  
            
        return actions