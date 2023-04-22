'''
Created on Dec 1, 2021

@author: cquzh
'''
from com.tao.py.rl.data.TrainDataItem import TrainDataItem
from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.rl.event.IDecisionMadeEvent import IDecisionMadeSimEvent


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
        #self.preReward=0
        self.environment=environment
    
    def onEventTriggered(self,event): 
        if isinstance(event, IDecisionMadeSimEvent):
            self.onDecisionMade(event)
            
        
    def onDecisionMade(self):
        state=self.environment.getCurState()
        if self.preState!=None:            
            item=TrainDataItem(self.preState,self.preAction,self.environment.getReward(),state,self.environment.getCurActualActionSet())
            self.dataset.append(item)
        
        self.preState=state
        self.preAction=self.environment.getAction()        
        #self.preReward=self.environment.getReward()
                
    def getDataset(self):
        return self.dataset
    
    def __str__(self):
        return "\n".join(map(str,[item.__str__() for item in self.dataset]))
    
    def flatten(self):
        data=[]
        data.extends([item.flatten() for item in self.dataset])
        return data
    