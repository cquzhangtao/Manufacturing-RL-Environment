'''
Created on Dec 24, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.manu.event.DecisionMakingEvent import DecisionMakingEvent
from com.tao.py.manu.event.JobReleaseEvent import JobReleaseEvent
from com.tao.py.manu.event.JobDepartureEvent import JobDepartureEvent
from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent

class WIPReward(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.preWIP=0
        self.preWIPChangeTime=0
        self.totalWIP=0
    def reset(self):
        self.preWIP=0
        self.preWIPChangeTime=0
        self.totalWIP=0

        
    def onEventTriggered(self,event): 
        curTime=event.getTime()
        if isinstance(event, DecisionMadeEvent):
            self.totalWIP+=(curTime-self.preWIPChangeTime)*self.preWIP
            self.preWIPChangeTime=curTime          
            
        elif isinstance(event,JobReleaseEvent):
            self.totalWIP+=(curTime-self.preWIPChangeTime)*self.preWIP
            self.preWIPChangeTime=curTime
            self.preWIP+=event.job.originProcessTime
        elif isinstance(event,JobDepartureEvent):
            self.totalWIP+=(curTime-self.preWIPChangeTime)*self.preWIP
            self.preWIPChangeTime=curTime
            self.preWIP-=event.job.originProcessTime
            
    def getReward(self,scenario,replication,model,tool,queue,job,time): 
        self.totalWIP+=(time-self.preWIPChangeTime)*self.preWIP
        self.preWIPChangeTime=time 
        reward=2000-self.totalWIP
        #print(str(time)+","+str(reward))
        self.totalWIP=0
        return  reward     