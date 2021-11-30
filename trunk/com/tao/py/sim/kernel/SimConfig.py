'''
Created on Nov 30, 2021

@author: cquzh
'''

class SimConfig(object):
    '''
    classdocs
    '''


    def __init__(self, simLen,rep):
        '''
        Constructor
        '''
        self.simLen=simLen
        self.replication=rep
        
    def getSimLen(self):
        return self.simLen