'''
Created on Apr 22, 2023

@author: xiesh
'''
from com.tao.py.sim.kernel.SimEventListener import SimEventListener

class ISimResult(SimEventListener):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def reset(self):
        pass
    
    def getKPI(self):   
        pass
    
    def summarize(self):
        pass 
    
    def toString(self):
        pass