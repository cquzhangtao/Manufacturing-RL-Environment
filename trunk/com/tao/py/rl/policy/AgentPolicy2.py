'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from com.tao.py.rl.kernel.Action import Action
class AgentPolicy2(object):
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

        '''
        actual actions q value
        '''
        pairs=[(state.getData()[0],action.getData()[0]) for action in actions]
        qvalues=self.agent.eval(pairs)
        maxIdx=qvalues.index(max(qvalues))
            
     
        return maxIdx,actionIdices[maxIdx]