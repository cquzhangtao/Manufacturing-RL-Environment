'''
Created on Dec 4, 2021

@author: Shufang
'''


from com.tao.py.rl.environment.Environment3 import SimEnvironment3
from com.tao.py.rl.kernel.Action import Action
import itertools



class SimEnvironment4(SimEnvironment3):

    def __init__(self,scenario,name=""):
        super().__init__(scenario,name=name)
        self.init(10)
        self.actionFeatureDiscretSize=10
        self.featureSplitSize,self.actionNum=self.calActionNum()
        self.allactions=list(itertools.product(* self.featureSplitSize))
    
    def calActionNum(self):
        envSpec=self.environmentSpec
        actionFeatureNum=envSpec.actionFeatureNum
        
        featureSplitSize=[] 
        count=1
        for idx in range(actionFeatureNum):
            featureSplitSize.append([])
            amax=envSpec.maxAction[idx]
            amin=envSpec.minAction[idx]
            if amax==amin:
                featureSplitSize[idx].extend(0)
                count*=1
            else:
                featureSplitSize[idx].extend(range(self.actionFeatureDiscretSize))
                count*=self.actionFeatureDiscretSize
        
        return featureSplitSize,count
    
    def getJobByIndex(self,actionIdx):
        #actionIdices=[feature for action in self.actions for feature in action.getData()]
        if actionIdx not in self.actionIdices:
            a=0
        queueIdx=self.actionIdices.index(actionIdx)
        return self.queue[queueIdx]
    
    def getQueueIdx(self,actionIdx):
        return self.actionIdices.index(actionIdx)
    
    def updateCurrentState(self):
        super().updateCurrentState()
        self.actionIdices=[feature for action in self.actions for feature in action.getData()]
    
    def getMask(self):
        
        mask=[]
        for i in range(self.actionNum):
            if i in self.actionIdices:
                mask.append(1)
            else:
                mask.append(0)
        
        return mask
            
    
    def getActionIndex(self,action):
        envSpec=self.environmentSpec
        actionFeatureNum=envSpec.actionFeatureNum 
             
        idx=0
        featureSplitPos=[0] * actionFeatureNum
        for feature in action.getData():
            if len(self.featureSplitSize[idx])==1:
                continue
            
            amin=envSpec.minAction[idx] 
            amax=envSpec.maxAction[idx]
            avalue=feature 
            if avalue<amin:                
                featureSplitPos[idx]=0
            elif avalue>=amax:
                featureSplitPos[idx]=self.actionFeatureDiscretSize-1
            else:
                fstep=(amax-amin)/(self.actionFeatureDiscretSize-2)
                featureSplitPos[idx]= int(avalue//fstep) +1              
            
            idx+=1
            
        actionIdx=self.allactions.index(tuple(featureSplitPos))
        
        return actionIdx
        
             
    
    
    def getActionFromJob(self,job,time):
        action= super().getActionFromJob(job, time)
        if self.initializing:
            return action

        idx=self.getActionIndex(action)
        return Action([idx])
    
