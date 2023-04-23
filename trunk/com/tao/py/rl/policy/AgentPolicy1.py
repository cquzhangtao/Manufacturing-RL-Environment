'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from com.tao.py.rl.kernel.Action import Action
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
        maxIdx=0
        idx=-1
        
        '''
        all actions q values
        '''
        qvalues=self.agent.calQValue(state.getData()) 
        for qvalue in qvalues[0]:
            idx+=1
            if idx not in  actionIdices:
                continue
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=idx
            
     
        return actionIdices.index(maxIdx),maxIdx