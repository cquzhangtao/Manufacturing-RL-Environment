'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity

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
        
    def getModel(self):
        return self.model
    
    def getEngine(self):
        return self.engine
    
    def setModel(self,model):
        self.model=model
        
    def setEngine(self,engine):
        self.engine=engine
    
    def addEvent(self,event):
        self.engine.insertEvent(event)
        
    def removeEvent(self,event):
        self.engine.eventList.remove(event)
        
    def getEvent(self):
        return self.event
    
    def setEvent(self,event):
        self.event=event