'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
class AgentPolicy1(object):
    '''
    classdocs
    '''


    def __init__(self, agent,epsilon):
        '''
        Constructor
        '''
        self.agent=agent
        self.epsilon=epsilon
        self.environment=agent.environment
        
    def getAction(self,state,actions):
        
        #actions=[Action([self.environment.getActionIndex(action)]) for action in actions]
        actionIdices=[feature for action in actions for feature in action.getData() ]
        prob=random.random()
        if prob<self.epsilon:
            idx=random.randint(0, len(actionIdices)-1)
            return idx,actionIdices[idx]
                
        maxQ=float('-inf')
        '''
        all actions q values
        '''            
        maxIdx=[]
        idx=-1
        qvalues=self.agent.calQValue(state.getData()) 
        for qvalue in qvalues[0]:
            idx+=1
            if idx not in  actionIdices:
                continue
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=[]
                maxIdx.append(idx)
            elif abs(qvalue-maxQ)<0.000000001:
                maxIdx.append(idx)
        maxIdx=maxIdx[random.randint(0,len(maxIdx)-1)]
        #maxIdx=maxIdx[len(maxIdx)-1]
        return actionIdices.index(maxIdx),maxIdx