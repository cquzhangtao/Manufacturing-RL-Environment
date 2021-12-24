'''
Created on Dec 1, 2021

@author: cquzh
'''
from com.tao.py.rl.data.TrainDataItem import TrainDataItem
from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent


class TrainDataCollector(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self,environment):
        '''
        Constructor
        '''
        self.dataset=[]
        #self.result=result
        self.preState=None
        self.preAction=None
        self.preReward=0
        self.environment=environment
    
    def onEventTriggered(self,event): 
        if isinstance(event, DecisionMadeEvent):
            self.onDecisionMade(event,event.getJob(),event.getTool(),event.getQueue(),event.getTime())
            
        
    def onDecisionMade(self,event,job,tool,queue,time):
        state=self.environment.getStateFromModel(job.getModel(), tool,queue,time)
        if self.preState!=None:            
            item=TrainDataItem(self.preState,self.preAction,self.preReward,state,self.environment.getActionSetFromQueue(queue,time))
            self.dataset.append(item)
        
        self.preState=state
        self.preAction=self.environment.getActionFromJob(job,time)        
        self.preReward=self.environment.getReward(event.getScenario().getIndex(),event.getReplication(),job.getModel(),tool,queue,job,time)

                
    def getDataset(self):
        return self.dataset
    
    def __str__(self):
        return "\n".join(map(str,[item.__str__() for item in self.dataset]))
    
    def flatten(self):
        data=[]
        data.extends([item.flatten() for item in self.dataset])
        return data
    