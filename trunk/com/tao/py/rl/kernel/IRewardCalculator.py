'''
Created on Apr 22, 2023

@author: xiesh
'''
from com.tao.py.sim.kernel.SimEventListener import SimEventListener

class IRewardCalculator(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def reset(self):
        pass

            
    def getReward(self,env): 
        return 0  