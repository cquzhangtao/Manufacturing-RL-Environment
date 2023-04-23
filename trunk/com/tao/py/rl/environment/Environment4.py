'''
Created on Dec 4, 2021

@author: Shufang
'''


from com.tao.py.rl.environment.Environment3 import SimEnvironment3
from com.tao.py.rl.kernel.Action import Action
import itertools




class SimEnvironment4(SimEnvironment3):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name=""):
        super().__init__(scenario,resultContainerFn,rewardCalculatorFn=rewardCalculatorFn,name=name,init_runs=200)
        #self.init(10)
        self.actionFeatureDiscretSize=3
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
            idenNum=envSpec.countAction[idx]
            if amax==amin:
                featureSplitSize[idx].extend(range(1+2))
                count*=1+2
            elif idenNum<self.actionFeatureDiscretSize:
                featureSplitSize[idx].extend(range(idenNum+2))
                count*=idenNum+2        
            
            else:
                featureSplitSize[idx].extend(range(self.actionFeatureDiscretSize+2))
                count*=self.actionFeatureDiscretSize+2
        
        return featureSplitSize,count
    
    #def getJobByIndex(self,actionIdx):
        #actionIdices=[feature for action in self.actions for feature in action.getData()]
    #    if actionIdx not in self.actionIdices:
    #       a=0
    #   queueIdx=self.actionIdices.index(actionIdx)
    #   return self.queue[queueIdx]
    
    def getIdxInActualActionSet(self,actionIdx):
        idx=self.actionIdices.index(actionIdx)
        #print(idx)
        return idx
    
    def updateCurrentState(self):
        super().updateCurrentState()
        self.actionIdices=[feature for action in self.getActions() for feature in action.getData()]
    
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
            flen=len(self.featureSplitSize[idx])
            amin=envSpec.minAction[idx] 
            amax=envSpec.maxAction[idx]
            idenNum=envSpec.countAction[idx]
            uList=envSpec.uniqueAction[idx]
            avalue=feature 
            if avalue<amin:                
                featureSplitPos[idx]=0
            elif avalue==amin:
                featureSplitPos[idx]=1
            elif avalue==amax:
                featureSplitPos[idx]=flen-2
            elif avalue>amax:
                featureSplitPos[idx]=flen-1
            elif idenNum<self.actionFeatureDiscretSize:
                featureSplitPos[idx]=uList.index(avalue)+1
            else:
                fstep=(amax-amin)/(flen-2)
                iidx=0
                start=amin
                end=start+fstep
                while True:
                    if avalue>=start and avalue<end:
                        break
                    iidx+=1
                    start=end
                    end=start+fstep
                    

                featureSplitPos[idx]= iidx+1             
            
            idx+=1
            
        actionIdx=self.allactions.index(tuple(featureSplitPos))
        
        #print(str(action)+" "+str(actionIdx))
        
        return actionIdx
        
             
    
    
    def getAction(self):
        action= super().getAction()
        if self.initializing:
            return action

        idx=self.getActionIndex(action)
        return Action([idx])
    
    def getActions(self):
        actions= super().getActions()
        if self.initializing:
            return actions

        return [Action([self.getActionIndex(action)]) for action in actions]