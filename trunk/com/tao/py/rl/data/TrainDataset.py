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
        self.dataCollector=dataCollector
        self.discount=0.1        
        self.rawData=dataCollector.flatten()

        self.getDataSpec()
        self.calPreProcParameters()
        
        self.input=[row[self.stateIdx:self.rewardIdx] for row in self.rawData]
        
        self.normalizedInput=self.normalizeInput()
        
        #self.getTrainData()
    
    def getInputSize(self): 
        return  self.actionFeatureNum+ self.stateFeatureNum
        
        
    def getDataSpec(self): 
        item=self.dataCollector.getOneTrainDataItem()
        
        self.actionFeatureNum =len(item.getAction().getData())
        self.stateFeatureNum=len(item.getState().getData())
        self.stateIdx=0
        self.actionIdx=self.stateIdx+self.stateFeatureNum
        self.rewardIdx=self.actionIdx+self.actionFeatureNum
        self.newStateIdx=self.rewardIdx+1
        self.newActionSetIdx=self.newStateIdx+self.stateFeatureNum  
        
    def normalizeInput(self):
        #return [(row[idx]-self.min[idx])/(self.max[idx]-self.min[idx]) for row in self.input for idx in range(len(row))]
        return [(row-self.min)/(self.max-self.min) for row in self.input ]
    
    def getNormalizedInput(self):  
        return self.normalizedInput  
    
    def calPreProcParameters(self):
        allStateData=[row[self.stateIdx:self.actionIdx] for row in self.rawData]
        allStateData.extend([row[self.newStateIdx:self.newActionSetIdx] for row in self.rawData])
        
        allActionData=[row[self.actionIdx:self.rewardIdx] for row in self.rawData]
        allActionData.extend([row[col:col+self.actionFeatureNum] for row in self.rawData for col in range(self.newActionSetIdx,len(row),self.actionFeatureNum) ])       
        
        #TODO remove duplication
        
        varState=numpy.var(allStateData,axis=0)
        varAction=numpy.var(allActionData,axis=0)
        self.var=numpy.concatenate((varState,varAction))
        
        meanState=numpy.mean(allStateData,axis=0)
        meanAction=numpy.mean(allActionData,axis=0)
        self. mean=numpy.concatenate((meanState,meanAction))
        
        minState=numpy.min(allStateData,axis=0)
        minAction=numpy.min(allActionData,axis=0)
        self. min=numpy.concatenate((minState,minAction))
        
        maxState=numpy.max(allStateData,axis=0)
        maxAction=numpy.max(allActionData,axis=0)
        self. max=numpy.concatenate((maxState,maxAction))
    
    def calTargetQValue(self,reward,nextState,nextActions):
        qvalues=([self.calQValue(nextState,nextActions[col:col+self.actionFeatureNum]) for col in range(0,len(nextActions),self.actionFeatureNum) ])
        return reward+self.discount*max(qvalues)
    
    def calQValue(self,state,action):
        return 0
    
    def getOutputTarget(self):
        #trainData=[row[self.stateIdx:self.rewardIdx]+[self.calTargetQValue(row[self.rewardIdx],row[self.newStateIdx:self.newActionSetIdx],row[self.newActionSetIdx:len(row)])] for row in self.rawData]
        #trainData=[ numpy.concatenate((self.normalizedInput[row],[self.calTargetQValue(self.rawData[row][self.rewardIdx],self.rawData[row][self.newStateIdx:self.newActionSetIdx],self.rawData[row][self.newActionSetIdx:len(self.rawData[row])])])) for row in range(len(self.normalizedInput)) ]
        #trainData=[[self.rawData[row][self.rewardIdx]]for row in range(len(self.normalizedInput)) ]
        output=[[self.calTargetQValue(self.rawData[row][self.rewardIdx],self.rawData[row][self.newStateIdx:self.newActionSetIdx],self.rawData[row][self.newActionSetIdx:len(self.rawData[row])])] for row in range(len(self.normalizedInput)) ]
        return output
