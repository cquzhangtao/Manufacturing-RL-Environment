'''
Created on Apr 22, 2023

@author: xiesh
'''

from com.tao.py.sim.kernel.SimEvent import SimEvent
class IDecisionMadeSimEvent(SimEvent):
    '''
    classdocs
    '''


    def __init__(self, time,priority):
        '''
        Constructor
        '''
        super().__init__(time,priority)

        
    
    def getState(self,env):
        pass
    def getAction(self,env):
        pass  
    def getReward(self,env):
        pass  
    def getActions(self,env):
        pass
    
