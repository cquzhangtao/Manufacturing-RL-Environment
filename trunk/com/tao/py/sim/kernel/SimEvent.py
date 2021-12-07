'''
Created on Nov 29, 2021

@author: Shufang
'''

from com.tao.py.sim.kernel.SimEntity import SimEntity

class SimEvent(SimEntity):
    '''
    classdocs
    '''


    def __init__(self, time,priority):
        '''
        Constructor
        '''
        super().__init__(0, "",None)
        self.time=time
        self.priority=priority
        
    def trigger(self):
        pass
    
    def getTime(self):
        return self.time
    
    def getPriority(self):
        return self.priority
        