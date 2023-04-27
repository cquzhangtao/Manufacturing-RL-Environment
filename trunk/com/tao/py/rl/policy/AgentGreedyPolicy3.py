'''
Created on Dec 6, 2021

@author: Shufang
'''
import random

class AgentGreedyPolicy3(object):
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
        
        probabilities=[self.agent.paiAS[state.getData()[0]][action.getData()[0]] for action in actions]
        maxProb=max(probabilities)
        indices=[idx for idx in range(len(probabilities)) if abs(probabilities[idx]-maxProb)<0.00000001]
        #idx=indices[random.randint(0,len(indices)-1)]
        idx=indices[len(indices)-1]
        return idx,actions[idx].getData()[0]
