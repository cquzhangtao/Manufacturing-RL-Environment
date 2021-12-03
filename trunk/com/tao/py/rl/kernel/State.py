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
        
    def getData(self):
        return self.data
    
    def __str__(self):
        return ",".join([ "{:.2f}".format(value) for value in self.data])   