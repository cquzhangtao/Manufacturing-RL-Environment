'''
Created on Dec 1, 2021

@author: cquzh
'''

from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.rl.event.IDecisionMadeEvent import IDecisionMadeSimEvent
from com.tao.py.rl.data.TrainDataCollector import TrainDataCollector


class TrainDataCollectors(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self,environment):
        '''
        Constructor
        '''
        self.dataset=[]
        #self.resut=result
        self.collectors=[]
        
        self.environment=environment

    
    def onEventTriggered(self,event): 
        scenario=0
        rep=0
        if isinstance(event, IDecisionMadeSimEvent):
            scenario=event.getScenario().getIndex()
            rep=event.getReplication()
            self.extend2DArray(self.collectors, scenario, rep)
            
            if self.collectors[scenario][rep]==None:
                self.collectors[scenario][rep]=TrainDataCollector(self.environment)
                self.dataset.append(self.collectors[scenario][rep].getDataset())
            subCollector=self.collectors[scenario][rep]
            subCollector.onDecisionMade()
                
    def getDataset(self): 
        return  self.dataset  

    def __str__(self):
        return "\n".join(map(str,[repData.__str__() for sceData in self.dataset for repData in sceData]))
    
    def flatten(self):
        data=[]
        data.extend([repData.flatten() for sceData in self.dataset for repData in sceData])
        return data
