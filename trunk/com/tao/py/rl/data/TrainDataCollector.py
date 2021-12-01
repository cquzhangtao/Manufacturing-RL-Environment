'''
Created on Dec 1, 2021

@author: cquzh
'''
from com.tao.py.rl.data.TrainDataItem import TrainDataItem
from com.tao.py.rl.kernel.State import State
from com.tao.py.rl.kernel.Action import Action
from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent


class TrainDataCollector(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self,result):
        '''
        Constructor
        '''
        self.dataset=[]
        self.result=result
        self.preState=None
        self.preAction=None
        self.preReward=0
    
    def onEventTriggered(self,event): 
        if isinstance(event, DecisionMadeEvent):
            self.onDecisionMade(event,event.getJob(),event.getTool(),event.getQueue(),event.getTime())
            
        
    def onDecisionMade(self,event,job,tool,queue,time):
        state=self.getStateFromModel(job.getModel(), tool,queue,time)
        if self.preState!=None:            
            item=TrainDataItem(self.preState,self.preAction,self.preReward,state,self.getActionSetFromQueue(queue,time))
            self.dataset.append(item)
        
        self.preState=state
        self.preAction=self.getActionFromJob(job,time)        
        self.preReward=self.getReward(event.getScenario().getIndex(),event.getReplication())
            
    
    def getStateFromModel(self,model,tool,queue,time):
        return State([time,len(queue)])
    
    def getActionFromJob(self,job,time): 
        return Action([job.getProcessTime(),time-job.getReleaseTime()])
       
    def getActionSetFromQueue(self,queue,time):  
        actions=[]
        for job in queue:
            actions.append(self.getActionFromJob(job,time))  
            
        return actions
    
    def getReward(self,scenario,replication):        
        return 1/self.result.getDataset(scenario,replication).getAvgCT()
    
    def getDataset(self):
        return self.dataset
    
    def __str__(self):
        return ",".join(map(str,self.dataset))
    