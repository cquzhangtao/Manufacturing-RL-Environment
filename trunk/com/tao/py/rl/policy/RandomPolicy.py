'''
Created on Dec 6, 2021

@author: Shufang
'''
import random
from com.tao.py.rl.policy.SimplePolicy import SimplePolicy

class RandomPolicy(SimplePolicy):
    '''
    classdocs
    '''
    def __init__(self,env):
        '''
        Constructor
        '''
        self.environment=env
        
    def getAction(self,state,actions):
        return random.randint(0,len(actions)-1)     
    