'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity

scenarioIndex=0

class Scenario(Entity):
    '''
    classdocs
    '''


    def __init__(self, uuid,name, simConfig, createModelFn):
        '''
        Constructor
        '''
        super().__init__(uuid, name)
        self.simConfig=simConfig
        self.createModelFn=createModelFn
        global scenarioIndex
        self.index=scenarioIndex
        scenarioIndex+=1
        #self.rule=rule
        
    def getSimConfig(self):
        return self.simConfig
    
    def createModel(self):
        return self.createModelFn()
    
        