'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
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
        self.epsilonDecay=ExponentialDecay( epsilon,decay_steps=1000,decay_rate=0.96)
        self.step=0
    def getAction(self,state,actions):
        self.step+=1
        #actions=[Action([self.environment.getActionIndex(action)]) for action in actions]
        actionIdices=[feature for action in actions for feature in action.getData() ]
        prob=random.random()
        #print(self.epsilonDecay(self.step))
        if prob<self.epsilonDecay(self.step):
            idx=random.randint(0, len(actionIdices)-1)
            return idx,actionIdices[idx]

        '''
        actual actions q value
        '''
        pairs=[(state.getData()[0],action.getData()[0]) for action in actions]
        qvalues=self.agent.eval(pairs)
        maxIdx=qvalues.index(max(qvalues))
            
     
        return maxIdx,actionIdices[maxIdx]