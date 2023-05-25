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
    '''
    fixed number of states and actions
    states and actions are described by their indices
    '''

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=100):
        super().__init__(scenario,resultContainerFn,rewardCalculatorFn=rewardCalculatorFn,name=name,init_runs=init_runs)
        #self.init(10)
        
        if init_runs>0:
            self.stateFeatureDiscretSize=10
            self.stateFeatureSplitSize,self.stateNum=self.calStateNum()
            self.allStates=list(itertools.product(* self.stateFeatureSplitSize))
            print("fixed states number"+str(len(self.allStates)));
            print(self.allStates)
            self.calStateCumProduct()
            
            
    def calStateCumProduct(self):
        self.stateCumProduct=[1]*len(self.stateFeatureSplitSize)
        for idx in range(len(self.stateFeatureSplitSize)-2,-1,-1):
            self.stateCumProduct[idx]=len(self.stateFeatureSplitSize[idx+1])*self.stateCumProduct[idx+1]
            
    
    
    
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
                featureSplitSize[idx].extend(range(1+0))
                count*=1+0
            elif idenNum<self.stateFeatureDiscretSize:
                featureSplitSize[idx].extend(range(idenNum+0))
                count*=idenNum+0        
            
            else:
                featureSplitSize[idx].extend(range(self.stateFeatureDiscretSize+0))
                count*=self.stateFeatureDiscretSize+0
        
        return featureSplitSize,count
            
    
    def getStateIndex(self,state):
        envSpec=self.environmentSpec
        stateFeatureNum=envSpec.stateFeatureNum 
             
        idx=0
        featureSplitPos=[0] * stateFeatureNum
        for feature in state.getData():
            if len(self.stateFeatureSplitSize[idx])==1:
                featureSplitPos[idx]=0
                idx+=1
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
                featureSplitPos[idx]=0
            elif avalue==amax:
                featureSplitPos[idx]=flen-1 
            elif avalue>amax:
                featureSplitPos[idx]=flen-1
            elif idenNum<self.stateFeatureDiscretSize:
                featureSplitPos[idx]=uList.index(avalue)+0
            else:
                fstep=(amax-amin)/(flen-0)
                iidx=0
                start=amin
                end=start+fstep
                while True:
                    if avalue>=start and avalue<end:
                        break
                    iidx+=1
                    start=end
                    end=start+fstep
                    

                featureSplitPos[idx]= iidx+0             
            
            idx+=1
            
        stateIdx=0
        
        idx=0
        for pos in featureSplitPos:
            stateIdx+=pos*self.stateCumProduct[idx]
            idx+=1
               
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
            pickle.dump(self.stateCumProduct, file)
            
            
    def loadSpec(self,path):
        self.environmentSpec=TrainDataset(None)
        with open(path, 'rb') as file:
            super().loadSpecInner(pickle,file)
            self.stateFeatureDiscretSize=pickle.load(file)
            self.stateFeatureSplitSize=pickle.load(file)
            self.stateNum=pickle.load(file)
            self.allStates=pickle.load(file)   
            self.stateCumProduct=pickle.load(file)         