'''
Created on Dec 6, 2021

@author: Shufang
'''

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
        idx=probabilities.index(max(probabilities))
     
        return idx,actions[idx].getData()[0]
