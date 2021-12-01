'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity
from pyexpat import model

class SimEntity(Entity):
    '''
    classdocs
    '''


    def __init__(self, uuid, name,model):
        '''
        Constructor
        '''
        super().__init__(uuid,name)
        
        self.model=model
        self.engine=None
        self.event=None
        self.replication=0
        self.scenario=None
        
    def copySimContext(self,entity):
        entity.engine=self.engine
        entity.scenario=self.scenario
        entity.replication=self.replication
    
    
    def getModel(self):
        return self.model
    
    def getEngine(self):
        return self.engine
    
    def setModel(self,model):
        self.model=model
        
    def setEngine(self,engine):
        self.engine=engine
    
    def addEvent(self,event):
        self.copySimContext(event)        
        self.engine.insertEvent(event)
        
    def removeEvent(self,event):
        self.engine.eventList.remove(event)
        
    def getEvent(self):
        return self.event
    
    def setEvent(self,event):
        self.event=event
        
    def getScenario(self):
        return self.scenario
    
    def getReplication(self):
        return self.replication
    
    def setScenario(self,scenario):
        self.scenario=scenario
    
    def setReplication(self,rep):
        self.replication=rep