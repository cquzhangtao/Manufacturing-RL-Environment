'''
Created on Dec 1, 2021

@author: Shufang
'''
import numpy 
class DataProcess(object):
    '''
    classdocs
    '''


    def __init__(self, dataCollector):
        '''
        Constructor
        '''
        self.dataCollector=dataCollector
        
        item=dataCollector.getOneTrainDataItem()
        
        actionFeatureNum =len(item.getAction().getData())
        stateFeatureNum=len(item.getState().getData())
        stateIdx=0
        actionIdx=stateIdx+stateFeatureNum
        rewardIdx=actionIdx+actionFeatureNum
        newStateIdx=rewardIdx+1
        newActionSetIdx=newStateIdx+stateFeatureNum

        
        
        rawData=dataCollector.flatten()
        
        allStateData=[row[stateIdx:actionIdx] for row in rawData]
        allStateData.extend([row[newStateIdx:newActionSetIdx] for row in rawData])
        allActionData=[row[actionIdx:rewardIdx] for row in rawData]
        #allActionData.extend([row[newActionSetIdx:newActionSetIdx+actionFeatureNum] for row in rawData])
        for i in range(newActionSetIdx,len(rawData[0]),actionFeatureNum):
            allActionData.extend([row[i:i+actionFeatureNum] for row in rawData])
        
       
       
        #mean=[sum(row[idx]) for row in allStateData for idx in range(len(row))]
        #print(allStateData)
        print(allActionData)
       
        varState=numpy.var(allStateData,axis=0)
        varAction=numpy.var(allActionData,axis=0)
        var=numpy.concatenate((varState,varAction))
        
        meanState=numpy.mean(allStateData,axis=0)
        meanAction=numpy.mean(allActionData,axis=0)
        mean=numpy.concatenate((meanState,meanAction))
        
        allStateData=0
        a=0