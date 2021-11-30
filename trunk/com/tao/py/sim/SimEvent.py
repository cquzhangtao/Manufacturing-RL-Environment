'''
Created on Nov 29, 2021

@author: Shufang
'''
from com.tao.py.utilities.Entity import Entity

class SimEvent(Entity):
    '''
    classdocs
    '''


    def __init__(self, time):
        '''
        Constructor
        '''
        self.time=time
        
    def trigger(self):
        pass
    
    def getTime(self):
        return self.time
        