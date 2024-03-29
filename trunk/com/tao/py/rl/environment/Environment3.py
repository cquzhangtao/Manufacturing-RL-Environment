'''
Created on Dec 4, 2021

@author: Shufang
'''


from com.tao.py.rl.environment.Environment2 import SimEnvironment2

from com.tao.py.sim.kernel.SimEventListener import SimEventListener
from com.tao.py.sim.rl.IJobDepartureEvent import IJobDepartureEvent
import tensorflow as tf


class SimEnvironment3(SimEnvironment2,SimEventListener):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=100):
        self.jobs=[]
        self.steps=[]
        self.supportNStep=True
        super().__init__(scenario,resultContainerFn,rewardCalculatorFn=rewardCalculatorFn,name=name,init_runs=init_runs)

    
    def clear(self):
        self.jobs=[]
        self.steps=[]
        super().clear()       
             
    def getSimEventListeners(self):
        return [self]+super().getSimEventListeners()        
    
    def onEventTriggered(self,event): 
        if self.supportNStep and isinstance(event, IJobDepartureEvent):
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
                ct+=self.jobs[idx].getFF()
                count+=1

            idx+=1  
        reward=count/ct
        
        idx=0
        for step in self.steps:
            if step<=start:
                idx+=1
            else:
                break
            
        del self.steps[0:idx]
        del self.jobs[0:idx]
        
        # if self.summaryWriter is None:
        #     self.summaryWriter=tf.summary
        #
        # self.summaryWriter.scalar("env/n_step_reward",reward,step=start)
            
        return reward          
