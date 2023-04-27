'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
import math
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
        self.epsilonDecay=ExponentialDecay( epsilon,decay_steps=2000,decay_rate=0.96)
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
        maxvalue=max(qvalues)
        maxIdx=[idx for idx in range(len(qvalues)) if math.isnan(qvalues[idx]- maxvalue)or abs(qvalues[idx]- maxvalue)<0.0000001]
        maxIdx=maxIdx[random.randint(0,len(maxIdx)-1)]    
        #maxIdx=maxIdx[len(maxIdx)-1]
        return maxIdx,actionIdices[maxIdx]