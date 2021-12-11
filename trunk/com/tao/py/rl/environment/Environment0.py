'''
Created on Dec 4, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.manu.stat.SimDataCollector import SimDataCollector
from com.tao.py.rl.data.TrainDataCollectors import TrainDataCollectors
from com.tao.py.rl.data.TrainDataset import TrainDataset
from com.tao.py.rl.kernel.State import State
from com.tao.py.rl.kernel.Action import Action
from com.tao.py.manu.rule.Rule import AgentRule, FIFORule, RandomRule
import matplotlib.pyplot as plt



class SimEnvironment0(object):

    def __init__(self,scenario):
        self.scenario=scenario
        self.state=None
        self.rep=0
        self.eventListeners=[]
        self.simResult=None
        self.kpi=[]
        self.rewards=[]
        self.environmentSpec=None
        self.initializing=False;
        self.init()
    
    def clear(self):
        self.rep=0        
        self.kpi=[] 
        self.rewards=[]   
        
    def getSimEventListeners(self):
        return []
    
    def start(self,training=False,rule=FIFORule()):
        self.simResult=SimDataCollector()
        self.eventListeners.append(self.simResult)
        self.eventListeners.extend(self.getSimEventListeners())
        
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
        self.rewards.append(0)
        self.sim.start(self.model)
        self.rep+=1
    


            
    def collectData(self,policy,rule=None,repNum=1): 
        self.eventListeners=[] 
        trainDataCollector=TrainDataCollectors(self) 
        self.eventListeners.append(trainDataCollector)

        if rule==None:
            rule=AgentRule(policy)
        
        for _ in range(repNum):
            self.start(training=False,rule=rule)
        


        trainDataset=TrainDataset(trainDataCollector)
        
        while len(trainDataset.rawData)==0:
            for _ in range(repNum):
                self.start(training=False,rule=rule)
            trainDataset=TrainDataset(trainDataCollector)   
            
        del  self.eventListeners[0]  
        
        self.kpi.append(self.simResult.getTotalSummary().getAvgCT())
        totalReward=sum([j for sub in trainDataset.reward for j in sub])
        print(self.simResult.getTotalSummary().toString()+"Total Reward:"+str(totalReward))
        self.rewards.append(totalReward)    
        
        return trainDataset
    
    def init(self,repNum=1):
        self.initializing=True
        self.environmentSpec=self.collectData(None, rule=RandomRule(),repNum=repNum)
        self.clear()
        self.initializing=False
        
    
    def getStateFromModel(self,model,tool,queue,time):
        return State([time,len(queue)])
    
    
    def getActionFromJob(self,job,time): 
        return Action([time,job.getProcessTime()])#,time-job.getReleaseTime()]#)
    
       
    def getActionSetFromQueue(self,queue,time):  
        actions=[]
        for job in queue:
            actions.append(self.getActionFromJob(job,time))  
            
        return actions
    
    def getReward(self,scenario,replication,model,tool,queue,job,time): 
        #return 10-time+job.getReleaseTime()       
        return 5-self.simResult.getReplicationSummary(scenario,replication).getAvgCT()
        #return 1/len(queue)
    
    def getRewardForStepByStep(self): 
        return self.getReward(self.scenario.getIndex(), self.rep-1, self.model, self.tool, self.queue, self.job, self.time)        

    
    def drawKPICurve(self): 
        _, ax1 = plt.subplots()
        ax1.plot(self.kpi)
        plt.title("Avg CT over replications")
        plt.xlabel("Replication")
        plt.ylabel("Avg CT")
        _, ax2 = plt.subplots()
        ax2.plot(self.rewards)
        plt.show()