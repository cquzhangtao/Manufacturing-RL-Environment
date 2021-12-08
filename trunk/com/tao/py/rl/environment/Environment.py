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


class SimEnvironment(PyEnvironment):

    def __init__(self,scenario):
        self.scenario=scenario
        self.state=None
        self.rep=1
        self.eventListeners=[]
        self.simResult=None

    
    def start(self,training=False,rule=FIFORule()):
        self.simResult=SimDataCollector()
        self.eventListeners.append(self.simResult)
        
        self.model=self.scenario.createModel()       
        self.model.setReplication(self.rep) 
        self.model.setScenario(self.scenario)
        self.model.training=training 
        
        for machine in self.model.machines:
            machine.rule=rule
            
        self.sim=Simulator(self.scenario.getSimConfig(),self.eventListeners)                
                
        for simEntity in self.model.getSimEntities():
            simEntity.setReplication(self.rep) 
            simEntity.setScenario(self.scenario)
            simEntity.training=training   
        
        self.sim.start(self.model)
    


            
    def collectData(self,policy): 
        self.eventListeners=[] 
        trainDataCollector=TrainDataCollectors(self) 
        self.eventListeners.append(trainDataCollector)

        self.start(training=False,rule=AgentRule(policy))
        
        print(self.simResult.getSummaryStr())
        trainDataset=TrainDataset(trainDataCollector)
        
        while len(trainDataset.rawData)==0:
            self.start(rule=AgentRule(policy))
            trainDataset=TrainDataset(trainDataCollector)   
            
        del  self.eventListeners[0]        
        
        return trainDataset
        
    
    def getStateFromModel(self,model,tool,queue,time):
        return State([time,len(queue)])
    
    
    def getActionFromJob(self,job,time): 
        return Action([job.getProcessTime(),time-job.getReleaseTime()])
    
    def getSpec(self):
        return 2,2
       
    def getActionSetFromQueue(self,queue,time):  
        actions=[]
        for job in queue:
            actions.append(self.getActionFromJob(job,time))  
            
        return actions
    
    def getReward(self,scenario,replication,model,tool,queue,job,time):        
        return 1/self.simResult.getDataset(scenario,replication).getAvgCT()
             
            
            