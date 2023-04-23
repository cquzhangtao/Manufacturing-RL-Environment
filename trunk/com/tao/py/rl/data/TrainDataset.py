'''
Created on Dec 1, 2021

@author: Shufang
'''
import numpy 

class TrainDataset(object):
    '''
    classdocs
    '''


    def __init__(self, dataCollector):
        '''
        Constructor
        '''
        if dataCollector==None:
            return
        self.dataCollector=dataCollector
        self.discount=0.9        
        self.rawData=dataCollector.flatten()
        
        if len(self.rawData)==0:
            return


        
        self.getDataSpec()
        self.calPreProcParameters()
        
        self.input=[row[self.stateIdx:self.rewardIdx] for row in self.rawData]
        self.state=[row[self.stateIdx:self.actionIdx] for row in self.rawData]
        self.action=[row[self.actionIdx:self.rewardIdx] for row in self.rawData]
        self.reward=[row[self.rewardIdx:self.newStateIdx] for row in self.rawData]
        self.newState=[row[self.newStateIdx:self.newActionSetIdx] for row in self.rawData]
        self.newActionSet=[row[self.newActionSetIdx:len(row)] for row in self.rawData]
        
        self.normalizedInput=self.normalizeInput()
        
        #self.getTrainData()
    def getSize(self):
        return len(self.rawData)
    

        
    def calNextMaxReward(self,agent,start,end):
        nextMaxReward=[]

        for rowIdx in range(start,end):

            row=self.newActionSet[rowIdx]
            state=self.newState[rowIdx]
            inputData=[]                   
            for col in range(0,len(row),self.actionFeatureNum):
                inputData.append(state+row[col:col+self.actionFeatureNum])
            qvalue=self.calQValueFromInput(agent,numpy.vstack(inputData))
            nextMaxReward.append([max(qvalue)])
            
        return nextMaxReward
    
    def calNextMaxRewardForOneStep(self,agent,state,actions):

        inputData=[]                   
        for action in actions:
            inputData.append(state.getData()+action.getData())
        qvalue=self.calQValueFromInput(agent,numpy.vstack(inputData))
        nextMaxReward=max(qvalue)
            
        return nextMaxReward
    
    def calNextMaxRewardForOneStepState(self,agent,state):
        state=self.normalizeState([state.getData()])
        qvalues=agent.calQValue(state)
        nextMaxReward=max(qvalues)
            
        return nextMaxReward
    
    def getOutputTarget(self,agent,start,end):
        nextMaxReward=self.calNextMaxReward(agent,start,end)
        qvalue=numpy.vstack(self.reward[start:end])+self.discount*numpy.vstack(nextMaxReward)
        #actQvalue=self.getActualOutput(agent)
        #qvalue=self.normalizeTargetOutput(qvalue)
        return qvalue
    
    def getOutputTargetForOneStep(self,agent,reward,newState,newActions,power=1):
        nextMaxReward=self.calNextMaxRewardForOneStep(agent,newState,newActions)
        qvalue=numpy.vstack(reward+(self.discount**power)*numpy.vstack(nextMaxReward))
        return qvalue
    
    def getOutputTargetForOneStepState(self,agent,reward,newState,power=1):
        nextMaxReward=self.calNextMaxRewardForOneStepState(agent,newState)
        qvalue=numpy.vstack(reward+(self.discount**power)*numpy.vstack(nextMaxReward))
        return qvalue
    
    def getOutputTargetForOneStepSaras(self,agent,reward,newState,newAction,power=1):
        nextReward=agent.calQValue(newState.getData(),newAction.getData())
        
        qvalue=numpy.vstack(reward+(self.discount**power)*numpy.vstack(nextReward))
        return qvalue
    
    def normalizeTargetOutput(self,target):
        minV=min(target)
        maxV=max(target)
        datalist=[(row-minV)/(maxV-minV) for row in target ] 
        
        return numpy.vstack(datalist) 
 
    
    def getActualOutput(self,agent):
        return self.calQValueFromInput(agent,self.getNormalizedInput())
        
    def getInputSize(self): 
        return  self.actionFeatureNum+ self.stateFeatureNum
        
        
    def getDataSpec(self): 
        sample=self.dataCollector.getDataset()[0][0]
        #self.stateFeatureNum,self.actionFeatureNum=self.dataCollector.getEnvironmentSpec()    
        self.stateFeatureNum=len(sample.getState().getData())
        self.actionFeatureNum=len(sample.getAction().getData()) 
        self.stateIdx=0
        self.actionIdx=self.stateIdx+self.stateFeatureNum
        self.rewardIdx=self.actionIdx+self.actionFeatureNum
        self.newStateIdx=self.rewardIdx+1
        self.newActionSetIdx=self.newStateIdx+self.stateFeatureNum  
        
    def normalizeInput(self):
        #return [(row[idx]-self.min[idx])/(self.max[idx]-self.min[idx]) for row in self.input for idx in range(len(row))]
        # if all(v == 0 for v in self.max-self.min):
        #     return numpy.vstack(self.input)  
        #
        # datalist=[(row-self.min)/(self.max-self.min) for row in self.input ] 
        
        #datalist=[(row-self.mean)/self.std for row in self.input ] 
        return self.normalize(self.input) 
    def normalize(self,listData):
        #datalist=[(row-self.mean)/self.std for row in listData ] 

        
        
        datalist=[2*item-1 for item in (row/self.max for row in listData) ] 
        return numpy.vstack(datalist) 
    
    def normalizeState(self,listData):

        #datalist=[(row-self.mean)/self.std for row in listData ] 
        datalist=[2*item-1 for item in (row/self.maxState for row in listData) ] 
        return numpy.vstack(datalist) 
    
    def normalizeOneStep(self,state,action):
        return self.normalize([state.getData()+action.getData()])
    
    def normalizeOneStepState(self,state):
        return self.normalizeState([state.getData()])
    
    def getNormalizedInput(self,start,end):  
        return self.normalizedInput[start:end]  
    
    def calPreProcParameters(self):
        allStateData=[row[self.stateIdx:self.actionIdx] for row in self.rawData]
        allStateData.extend([row[self.newStateIdx:self.newActionSetIdx] for row in self.rawData])
        
        allActionData=[row[self.actionIdx:self.rewardIdx] for row in self.rawData]
        allActionData.extend([row[col:col+self.actionFeatureNum] for row in self.rawData for col in range(self.newActionSetIdx,len(row),self.actionFeatureNum) ])       
        
        #TODO remove duplication
        if len(allStateData)==0 or len(allActionData)==0:
            a=1
        
        self.varState=numpy.var(allStateData,axis=0)
        self.varAction=numpy.var(allActionData,axis=0)
        self.var=numpy.concatenate((self.varState,self.varAction))
        
        self.meanState=numpy.mean(allStateData,axis=0)
        self.meanAction=numpy.mean(allActionData,axis=0)
        self.mean=numpy.concatenate((self.meanState,self.meanAction))
        
        self.minState=numpy.min(allStateData,axis=0)
        self.minAction=numpy.min(allActionData,axis=0)
        self.min=numpy.concatenate((self.minState,self.minAction))
        
        self.maxState=numpy.max(allStateData,axis=0)
        self.maxAction=numpy.max(allActionData,axis=0)
        self.max=numpy.concatenate((self.maxState,self.maxAction))
        
        self.stdState=numpy.std(allStateData,axis=0)
        self.stdAction=numpy.std(allActionData,axis=0)
        self.std=numpy.concatenate((self.stdState,self.stdAction))
        
        invertState=[list(x) for x  in zip(*allStateData)]
        invertAction=[list(x) for x  in zip(*allActionData)]
        self.countState=[len(set(row)) for row in invertState]
        self.countAction=[len(set(row)) for row in invertAction]
        self.uniqueAction=[list(set(row)) for row in invertAction]
        self.uniqueState=[list(set(row)) for row in invertState]
     
    def save(self,pickle,file):
        pickle.dump(self.minState,file)
        pickle.dump(self.maxState,file)        
        pickle.dump(self.minAction,file)
        pickle.dump(self.maxAction,file) 
        pickle.dump(self.min,file)
        pickle.dump(self.max,file)                      
        pickle.dump(self.countState,file)
        pickle.dump(self.countAction,file)  
        pickle.dump(self.uniqueAction,file)
        pickle.dump(self.uniqueState,file)
        pickle.dump(self.stateFeatureNum,file)    
        pickle.dump(self.actionFeatureNum,file)  
        
    def load(self,pickle,file):
        self.minState=pickle.load(file)
        self.maxState=pickle.load(file)        
        self.minAction=pickle.load(file)
        self.maxAction=pickle.load(file) 
        self.min=pickle.load(file)
        self.max=pickle.load(file)                      
        self.countState=pickle.load(file)
        self.countAction=pickle.load(file)  
        self.uniqueAction=pickle.load(file)
        self.uniqueState=pickle.load(file)  
        self.stateFeatureNum = pickle.load(file) 
        self.actionFeatureNum = pickle.load(file)         
    def calTargetQValue(self,agent,reward,nextState,nextActions):
        qvalues=([self.calQValue(agent,nextState,nextActions[col:col+self.actionFeatureNum]) for col in range(0,len(nextActions),self.actionFeatureNum) ])
        return reward+self.discount*max(qvalues)
    
    def calQValueFromInput(self,agent,inputData):
        inputData=self.normalize(inputData)
        return agent.eval(inputData)
    
    # def calQValue(self,agent,state,action):
    #     inputData=numpy.concatenate((state,action))
    #     inputData=self.normalize(inputData)
    #     return agent.eval(numpy.reshape(inputData,(1,len(inputData))))
    
    # def getOutputTarget(self,agent):
    #     #trainData=[row[self.stateIdx:self.rewardIdx]+[self.calTargetQValue(row[self.rewardIdx],row[self.newStateIdx:self.newActionSetIdx],row[self.newActionSetIdx:len(row)])] for row in self.rawData]
    #     #trainData=[ numpy.concatenate((self.normalizedInput[row],[self.calTargetQValue(self.rawData[row][self.rewardIdx],self.rawData[row][self.newStateIdx:self.newActionSetIdx],self.rawData[row][self.newActionSetIdx:len(self.rawData[row])])])) for row in range(len(self.normalizedInput)) ]
    #     #trainData=[[self.rawData[row][self.rewardIdx]]for row in range(len(self.normalizedInput)) ]
    #     output=numpy.vstack([self.calTargetQValue(agent,row[self.rewardIdx],row[self.newStateIdx:self.newActionSetIdx],row[self.newActionSetIdx:len(row)]) for row in self.rawData ])
    #     return output
