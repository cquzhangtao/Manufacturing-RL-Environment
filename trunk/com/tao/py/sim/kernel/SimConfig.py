'''
Created on Nov 30, 2021

@author: cquzh
'''

class SimConfig(object):
    '''
    classdocs
    '''


    def __init__(self, rep,simLen):
        '''
        Constructor
        '''
        self.simLen=simLen
        self.replication=rep
        
    def getSimLen(self):
        return self.simLen
    def getReplication(self):
        return self.replication