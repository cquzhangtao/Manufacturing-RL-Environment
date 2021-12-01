'''
Created on Dec 1, 2021

@author: cquzh
'''

class State(object):
    '''
    classdocs
    '''


    def __init__(self, data):
        '''
        Constructor
        '''
        self.data=data
    
    def __str__(self):
        return ",".join(map(str,self.data))  