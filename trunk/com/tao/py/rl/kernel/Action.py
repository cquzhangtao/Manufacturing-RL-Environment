'''
Created on Dec 1, 2021

@author: cquzh
'''

class Action(object):
    '''
    classdocs
    '''


    def __init__(self, data):
        '''
        Constructor
        '''
        self.data=data
        
    def getData(self):
        return self.data
    def __str__(self):
        return ",".join(map(str,self.data))  