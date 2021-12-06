'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent
import copy
from com.tao.py.rl.environment.DecisionEventListener import DecisionEventListener
from com.tao.py.manu.stat.SimDataCollector import SimDataCollector


class SimEnvironment(PyEnvironment):

    def __init__(self,scenario):
        self.scenario=scenario
        self.state=None
        self.rep=1
        self.eventListeners=[]

        self.start()
    
    
    def createEventListeners(self):
        self.decisionMaking=DecisionEventListener()
        self.eventListeners.append(self.decisionMaking)
        self.simResult=SimDataCollector()
        self.eventListeners.append(self.simResult)
    
    def start(self):
        self.createEventListeners()
        self.model=copy.deepcopy(self.scenario.getModel())       
        self.model.setReplication(self.rep) 
        self.model.setScenario(self.scenario)
        self.model.training=True 
            
        self.sim=Simulator(self.scenario.getSimConfig(),self.eventListeners)                
        self.model.setEngine(self.sim)
                
        for simEntity in self.model.getSimEntities():
            simEntity.setEngine(self.sim)
            simEntity.setReplication(self.rep) 
            simEntity.setScenario(self.scenario)
            simEntity.training=True   
        
        self.sim.start(self.model)
    
    def restart(self):
        self.rep+=1
        self.start()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self.restart()
        self.state=self.model.getState()
        return ts.restart(np.array([self.state], dtype=np.int32))
    
    def _step(self, action):
        if self.sim.getState()==3: 
            return self._reset()        
        
        event=DecisionMadeEvent(self.decisionMaking.time,self.decisionMaking.tool,1/self.simResult.getSummary().getAvgCT(),self.decisionMaking.queue)
        self.sim.insertEventOnTop(event)

        self.sim.resume()
        
        self.state=self.model.getState()
        reward=0
        
        if self.sim.getState()==3:  
            #return self._reset()      
            return ts.termination(np.array([self.state], dtype=np.int32), reward)
        else:
            return ts.transition(
              np.array([self.state], dtype=np.int32), reward=0.0, discount=1.0)
            
            
            
            