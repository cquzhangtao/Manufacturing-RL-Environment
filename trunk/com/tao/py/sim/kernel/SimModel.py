'''
Created on Nov 29, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.SimEntity import SimEntity

class SimModel(SimEntity):
    '''
    classdocs
    '''


    def __init__(self, uuid, name):
        '''
        Constructor
        '''
        super().__init__(uuid,name,self)
        self.simEntities=[]
        
    def getInitalEvents(self):
        pass
    
    def getSimEntities(self):
        return self.simEntities
    
    def addSimEntity(self,entity):   
        self.simEntities.append(entity)
    
        