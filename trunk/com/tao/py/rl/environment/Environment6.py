'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import array_spec


from com.tao.py.rl.environment.Environment3 import SimEnvironment3







class SimEnvironment6(SimEnvironment3,PyEnvironment):

    def __init__(self,scenario,rewardCalculator,name=""):
        super().__init__(scenario,rewardCalculator,name=name)

        envSpec=self.environmentSpec
        self.kpiNum=2
        self.maxActionNum=100
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum+2+self.maxActionNum*envSpec.actionFeatureNum+self.kpiNum,), dtype=np.float32, minimum=np.append(envSpec.minState,[0]*(self.maxActionNum*envSpec.actionFeatureNum+2+self.kpiNum)), maximum=np.append(envSpec.maxState,[1]*(self.maxActionNum*envSpec.actionFeatureNum+2+self.kpiNum)),name='observation')
        self._observation_spec_no_mask = array_spec.BoundedArraySpec(
            shape=(envSpec.stateFeatureNum,), dtype=np.float32, minimum=envSpec.minState, maximum=envSpec.maxState,name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(envSpec.actionFeatureNum,), dtype=np.float32, minimum=envSpec.minAction, maximum=envSpec.maxAction,name='action')
     
    
    def observation_spec(self) :
        return self._observation_spec

    def action_spec(self) :
        return self._action_spec  
    # @property
    # def batched(self) -> bool:
    #     return False
    # @property
    # def batch_size(self):
    #     return 0
    
    def restart(self):
        pass
    
    def getObservation(self):
        kpi=[]
        if self.finishedEpisode():
            if len(self.kpi)<1:
                kpi=[0,0]
            kpi=[self.kpi[len(self.kpi)-1],self.allEpisodTotalReward[len(self.allEpisodTotalReward)-1]]
        else:
            if len(self.kpi)<1:
                kpi=[0,0]
            else:
                kpi=[self.kpi[len(self.kpi)-1],self.allEpisodTotalReward[len(self.allEpisodTotalReward)-1]]
            
        end=self.maxActionNum
        if len(self.actions)<self.maxActionNum:
            end= len(self.actions)
        
        actions=[feature for action in self.actions[:end] for feature in action.getData()] 
        
        while len(actions)<self.maxActionNum*self.environmentSpec.actionFeatureNum:
            actions.append(0)
        
        return np.array(self.state.getData()+[len(self.actions),self.environmentSpec.actionFeatureNum]+actions+kpi, dtype=np.float32)
      
    def _reset(self): 
        self.start()  
        return ts.restart(self.getObservation())
    
    def getActionIndex(self,selaction):
        actionIdx=-1
        
        for action in self.actions:
            actionIdx+=1
            match=True
            for i, j in zip(selaction, action.getData()):
                if i != j:
                    match=False
                    break
            if match:
                return actionIdx
        return actionIdx
    
    def _step(self,action): 
        if self.finishedEpisode():
            return self._reset()
        actionIdx=self.getActionIndex(action)
        super().takeAction(actionIdx) 
        observ=self.getObservation() 
        if self.finishedEpisode() : 
            return ts.termination(observ, reward=self.reward)
         
        return ts.transition(observ, reward=self.reward, discount=0.5) 
    
