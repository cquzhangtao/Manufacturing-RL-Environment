'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
class AgentGreedyPolicy(object):
    '''
    classdocs
    '''


    def __init__(self, agent):
        '''
        Constructor
        '''
        self.agent=agent
        self.environment=agent.environment
        
    def getAction(self,state,actions):
        
        maxQ=float('-inf')
        maxIdx=0
        idx=0
        for action in actions:

            qvalue=self.agent.calQValue(state.getData(),action.getData()) 
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=idx
            idx+=1
     
        return maxIdx,maxIdx