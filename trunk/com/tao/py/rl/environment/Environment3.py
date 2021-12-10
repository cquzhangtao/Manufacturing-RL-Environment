'''
Created on Dec 4, 2021

@author: Shufang
'''


from com.tao.py.rl.environment.Environment2 import SimEnvironment2

from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.manu.event.JobDepartureEvent import JobDepartureEvent


class SimEnvironment3(SimEnvironment2,SimEventListener):

    def __init__(self,scenario):
        self.jobs=[]
        self.steps=[]
        super().__init__(scenario)

    
    def clear(self):
        self.jobs=[]
        self.steps=[]
        super().clear()       
             
    def getSimEventListeners(self):
        return [self]        
    
    def onEventTriggered(self,event): 
        if isinstance(event, JobDepartureEvent):
            self.jobs.append(event.getJob())
            self.steps.append(self.stepCounter)
    
    def getRewardInNStep(self, n): 
        return self.getRewardBetweenStep(self.stepCounter-n, self.stepCounter)
               
    def getRewardBetweenStep(self, start,end):
        
        idx=0
        ct=0
        count=0
        for step in self.steps:
            if step>=end:
                break
            if step>=start:
                ct+=self.jobs[idx].getCT()
                count+=1

            idx+=1  
        reward=count/ct
        
        idx=0
        for step in self.steps:
            if step==start:
                idx+=1
            else:
                break
            
        del self.steps[0:idx]
        del self.jobs[0:idx]
            
        return reward          