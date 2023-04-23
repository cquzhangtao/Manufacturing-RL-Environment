'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from com.tao.py.rl.kernel.Action import Action
class AgentGreedyPolicy1(object):
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
        actions=[Action([self.environment.getActionIndex(action)]) for action in actions]
        actionIdices=[feature for action in actions for feature in action.getData() ]
                
        maxQ=float('-inf')
        maxIdx=0
        idx=-1
        qvalues=self.agent.calQValue(state.getData()) 
        for qvalue in qvalues[0]:
            idx+=1
            if idx not in  actionIdices:
                continue
            if(qvalue>maxQ):
                maxQ=qvalue
                maxIdx=idx
            
     
        return actionIdices.index(maxIdx),maxIdx