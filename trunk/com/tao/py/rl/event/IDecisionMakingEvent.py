'''
Created on Apr 22, 2023

@author: xiesh
'''

from com.tao.py.sim.kernel.SimEvent import SimEvent
class IDecisionMakingSimEvent(SimEvent):
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
    def getActionSet(self,env):
        pass
    def createDecisionMadeEvent(self,selActionIdx):
        pass
