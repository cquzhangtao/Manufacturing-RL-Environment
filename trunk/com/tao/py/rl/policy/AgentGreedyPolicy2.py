'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from com.tao.py.rl.kernel.Action import Action
class AgentGreedyPolicy2(object):
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
        
        stateidx=state.getData()[0]#self.environment.getStateIndex(state)
        actionIdices=[action.getData()[0] for action in actions]

        
        #actions=[Action([self.environment.getActionIndex(action)]) for action in actions]
        #actionIdices=[feature for action in actions for feature in action.getData() ]
        '''
        actual actions q value
        '''
        pairs=[(stateidx,actionidx) for actionidx in actionIdices]
        qvalues=self.agent.eval(pairs)
        maxIdx=qvalues.index(max(qvalues))
            
     
        return maxIdx,actionIdices[maxIdx]