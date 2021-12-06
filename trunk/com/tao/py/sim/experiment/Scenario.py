'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity


class Scenario(Entity):
    '''
    classdocs
    '''


    def __init__(self, uuid,name, simConfig, model):
        '''
        Constructor
        '''
        super().__init__(uuid, name)
        self.simConfig=simConfig
        self.model=model
        
    def getSimConfig(self):
        return self.simConfig
    
    def createModel(self):
        return self.model()