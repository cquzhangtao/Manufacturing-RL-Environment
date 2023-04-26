'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
class AgentPolicy3(object):
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
        
        probabilities=self.agent.paiAS
        
        probOfActions=[]
        summ=0
        for action in actions:
            summ+=probabilities[state.getData()[0]][action.getData()[0]]
            probOfActions.append(summ)
        
        prob=random.uniform(0,summ)
        
        for idx in range(len(probOfActions)):
            if prob<=probOfActions[idx]:
                break            
     
        return idx,actions[idx].getData()[0]