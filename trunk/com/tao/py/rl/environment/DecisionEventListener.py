'''
Created on Dec 1, 2021

@author: cquzh
'''

from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.manu.event.DecisionMakingEvent import DecisionMakingEvent


class DecisionEventListener(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.tool=None
        self.queue=None
        self.time=0

    
    def onEventTriggered(self,event): 

        if isinstance(event, DecisionMakingEvent):
            self.tool=event.getTool()
            self.queue=event.getTool().getQueue()
            self.time=event.getTime()