'''
Created on Nov 29, 2021

@author: Shufang
'''
from com.tao.py.utilities.Entity import Entity

class SimEvent(Entity):
    '''
    classdocs
    '''


    def __init__(self, time,priority):
        '''
        Constructor
        '''
        self.time=time
        self.priority=priority
        
    def trigger(self):
        pass
    
    def getTime(self):
        return self.time
    
    def getPriority(self):
        return self.priority
        