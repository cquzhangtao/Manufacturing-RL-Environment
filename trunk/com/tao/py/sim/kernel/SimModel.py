'''
Created on Nov 29, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.SimEntity import SimEntity

class SimModel(SimEntity):
    '''
    classdocs
    '''


    def __init__(self, uuid, name,model,engine):
        '''
        Constructor
        '''
        super().__init__(uuid,name,model, engine)
        
    def getInitalEvents(self):
        pass
    
    def getSimEntities(self):
        pass