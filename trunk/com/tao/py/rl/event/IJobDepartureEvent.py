'''
Created on Nov 29, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.SimEvent import SimEvent

class IJobDepartureEvent(SimEvent):
    '''
    classdocs
    '''
    def __init__(self, time,priority):
        '''
        Constructor
        '''
        super().__init__(time,priority)
    
    def getJob(self):   
        pass