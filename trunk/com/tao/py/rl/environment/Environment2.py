'''
Created on Dec 4, 2021

@author: Shufang
'''
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
import numpy as np
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent
from com.tao.py.rl.environment.DecisionEventListener import DecisionEventListener
from com.tao.py.manu.stat.SimDataCollector import SimDataCollector
from com.tao.py.rl.data.TrainDataCollectors import TrainDataCollectors
from com.tao.py.rl.data.TrainDataset import TrainDataset
from com.tao.py.rl.kernel.State import State
from com.tao.py.rl.kernel.Action import Action
from com.tao.py.manu.rule.Rule import AgentRule, FIFORule
from com.tao.py.rl.environment.Environment import SimEnvironment
from com.tao.py.rl.data.TrainDataItem import TrainDataItem


class SimEnvironment2(SimEnvironment):

    def __init__(self,scenario):
        super().__init__(scenario)
    
    def _reset(self):
        self.rep+=1
        self.start()
        #return ts.restart(np.array([self.state], dtype=np.float32))
        
    def observation_spec(self) :
        return None

    def action_spec(self) :
        return None
    
    def start(self,training=True,rule=None):
        self.decisionMaking=DecisionEventListener()
        self.eventListeners.append(self.decisionMaking)
        if rule==None:
            rule=AgentRule(self.policy)
        super().start(training=training,rule=rule)
        self.updateCurrentState()
    
    
    def _step(self, actionIdx):
    
        event=DecisionMadeEvent(self.time,self.tool,self.queue[actionIdx],self.queue)
        self.tool.addEventOnTop(event)

        self.sim.resume()
        
        self.updateCurrentState()
        self.reward=1/self.simResult.getSummary().getAvgCT()
        
        if self.sim.getState()==3: 
            print(self.simResult.getSummaryStr())
            self._reset()    
        
        # if self.sim.getState()==3:  
        #     #return self._reset()      
        #     return ts.termination(np.array([self.state], dtype=np.float32), self.reward)
        # else:
        #     return ts.transition(
        #       np.array([self.state], dtype=np.float32), reward=self.reward, discount=1.0)
    
    
    def collectOneStepData(self): 

        actionIdx=self.policy.getAction(self.state,self.actions)
        trainData=TrainDataItem(self.state,self.actions[actionIdx],0,None,None)
        self._step(actionIdx)
        trainData.reward=self.reward
        
        trainData.nextState=self.state
        trainData.nextActions=self.actions
        
        return trainData
    
    
    def updateCurrentState(self):
        self.tool=self.decisionMaking.tool;
        self.queue=self.decisionMaking.queue;
        self.time=self.decisionMaking.time
        self.state=self.getStateFromModel(self.model,self.tool,self.queue,self.time)
        self.actions=self.getActionSetFromQueue(self.queue,self.time)
             
            
            