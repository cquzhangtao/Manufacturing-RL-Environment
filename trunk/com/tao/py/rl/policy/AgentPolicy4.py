'''
Created on Dec 6, 2021

@author: Shufang
'''
import random


class AgentPolicy4(object):
    '''
    classdocs
    '''


    def __init__(self, agent):
        '''
        Constructor
        '''
        self.agent=agent
        self.environment=agent.environment
    
    def getProb(self,state,actions,idx):
        pairs=[(state,action) for action in actions]
        qvalues=self.agent.eval(pairs)

        summ=0

        for qvalue in qvalues:
            summ+=qvalue
        return qvalues[idx]/summ
        
              
    
    def getAction(self,state,actions):

        actionIdices=[feature for action in actions for feature in action.getData() ]
        

        '''
        actual actions q value
        '''
        pairs=[(state.getData()[0],action.getData()[0]) for action in actions]
        qvalues=self.agent.eval(pairs)
        
        cumms=[]
        cumms.append(0)
        summ=0
        idx=0;
        for qvalue in qvalues:
            cumms.append(cumms[idx]+qvalue)
            summ+=qvalue
            idx+=1
        del cumms[0]
        
        prob=random.random()*summ
        idx=0;
        for cumm in cumms:
            if prob<=cumm:
                break
            idx+=1  
        
        return idx,actionIdices[idx]