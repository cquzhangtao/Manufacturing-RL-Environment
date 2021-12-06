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
        self.epsilon=epsilon
        
    def getAction(self,state,actions):
        prob=random.random()
        if prob<self.epsilon:
            return random.randint(0, len(actions)-1)
        
        
        maxQ=float('-inf')
        maxIdx=0
        idx=0
        for action in actions:

            qvalue=self.agent.calQValue(state.getData(),action.getData()) 
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=idx
            idx+=1
     
        return maxIdx