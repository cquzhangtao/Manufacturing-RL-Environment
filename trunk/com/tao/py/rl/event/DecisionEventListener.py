'''
Created on Dec 1, 2021

@author: cquzh
'''

from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.rl.event.IDecisionMadeEvent import IDecisionMadeSimEvent
from com.tao.py.rl.event.IDecisionMakingEvent import IDecisionMakingSimEvent



class DecisionEventListener(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self,environment):
        '''
        Constructor
        '''
        #self.tool=None
        #self.queue=None
        #self.time=0
        self.environment=environment
        self.decisionMakingEvent=None;
        self.decisionMadeEvent=None;
    
    def onEventTriggered(self,event): 

        if isinstance(event, IDecisionMadeSimEvent):
            self.decisionMadeEvent=event;
            self.environment.action=event.getAction()
    
    def beforeEventTriggered(self,event): 

        if isinstance(event, IDecisionMakingSimEvent):
            #self.tool=event.getTool()
            #self.queue=event.getTool().getQueue().copy()
            #self.time=event.getTime()
            self.decisionMakingEvent=event 
            self.environment.state= event.getState()
            self.environment.actions=event.getActions()  
                