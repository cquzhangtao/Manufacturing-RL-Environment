'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
class AgentPolicy(object):
    '''
    classdocs
    '''


    def __init__(self, agent,epsilon):
        '''
        Constructor
        '''
        self.agent=agent
        self.environment=agent.environment
        self.epsilon=epsilon
        
    def getAction(self,state,actions):
        prob=random.random()
        if prob<self.epsilon:
            idx=random.randint(0, len(actions)-1)
            return idx,idx
        
        
        maxQ=float('-inf')
        maxIdx=[]
        idx=0
        for action in actions:

            qvalue=self.agent.calQValue(state.getData(),action.getData()) 
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=[]
                maxIdx.append(idx)
            elif abs(qvalue-maxQ)<0.000000001:
                maxIdx.append(idx)

            idx+=1
        maxIdx=maxIdx[random.randint(0,len(maxIdx)-1)]  
        #maxIdx=maxIdx[len(maxIdx)-1]
        return maxIdx,maxIdx