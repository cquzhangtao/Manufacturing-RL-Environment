'''
Created on Dec 4, 2021

@author: Shufang
'''

import pickle
import itertools
from com.tao.py.rl.data.TrainDataset import TrainDataset
from com.tao.py.rl.environment.Environment4 import SimEnvironment4
from com.tao.py.rl.kernel.State import State



class SimEnvironment8(SimEnvironment4):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=200):
        super().__init__(scenario,resultContainerFn,rewardCalculatorFn=rewardCalculatorFn,name=name,init_runs=init_runs)
        #self.init(10)
        
        if init_runs>0:
            self.stateFeatureDiscretSize=3
            self.stateFeatureSplitSize,self.stateNum=self.calStateNum()
            self.allStates=list(itertools.product(* self.stateFeatureSplitSize))
            print("fixed states number"+str(len(self.allStates)));
            print(self.allStates)
            
    def calStateNum(self):
        envSpec=self.environmentSpec
        stateFeatureNum=envSpec.stateFeatureNum
        
        featureSplitSize=[] 
        count=1
        for idx in range(stateFeatureNum):
            featureSplitSize.append([])
            amax=envSpec.maxState[idx]
            amin=envSpec.minState[idx]
            idenNum=envSpec.countState[idx]
            if amax==amin:
                featureSplitSize[idx].extend(range(1+2))
                count*=1+2
            elif idenNum<self.stateFeatureDiscretSize:
                featureSplitSize[idx].extend(range(idenNum+2))
                count*=idenNum+2        
            
            else:
                featureSplitSize[idx].extend(range(self.stateFeatureDiscretSize+2))
                count*=self.stateFeatureDiscretSize+2
        
        return featureSplitSize,count
            
    
    def getStateIndex(self,state):
        envSpec=self.environmentSpec
        stateFeatureNum=envSpec.stateFeatureNum 
             
        idx=0
        featureSplitPos=[0] * stateFeatureNum
        for feature in state.getData():
            if len(self.stateFeatureSplitSize[idx])==1:
                continue
            flen=len(self.stateFeatureSplitSize[idx])
            amin=envSpec.minState[idx] 
            amax=envSpec.maxState[idx]
            idenNum=envSpec.countState[idx]
            uList=envSpec.uniqueState[idx]
            avalue=feature 
            if avalue<amin:                
                featureSplitPos[idx]=0
            elif avalue==amin:
                featureSplitPos[idx]=1
            elif avalue==amax:
                featureSplitPos[idx]=flen-2 
            elif avalue>amax:
                featureSplitPos[idx]=flen-1
            elif idenNum<self.stateFeatureDiscretSize:
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
            
        stateIdx=self.allStates.index(tuple(featureSplitPos))
        
        
        return stateIdx
        
    def adaptState(self,state):  
        if self.initializing:
            return state

        idx=self.getStateIndex(state)
        return State([idx])               
    
    
    def getState(self):
        state= super().getState()
        if self.initializing:
            return state

        idx=self.getStateIndex(state)
        return State([idx])
    
    
    def saveSpec(self,path): 
        with open(path, 'wb') as file:
            super().saveSpecInner(pickle,file)
            pickle.dump(self.stateFeatureDiscretSize, file)
            pickle.dump(self.stateFeatureSplitSize, file)
            pickle.dump(self.stateNum, file)
            pickle.dump(self.allStates, file)
            
            
    def loadSpec(self,path):
        self.environmentSpec=TrainDataset(None)
        with open(path, 'rb') as file:
            super().loadSpecInner(pickle,file)
            self.stateFeatureDiscretSize=pickle.load(file)
            self.stateFeatureSplitSize=pickle.load(file)
            self.stateNum=pickle.load(file)
            self.allStates=pickle.load(file)            