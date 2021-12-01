'''
Created on Dec 1, 2021

@author: cquzh
'''



class TrainDataItem(object):
    '''
    classdocs
    '''


    def __init__(self, state,action,reward,nextState,nextActions):
        '''
        Constructor
        '''
        self.state=state
        self.action=action
        self.reward=reward
        self.nextActions=nextActions
        self.nextState=nextState
        
    def __str__(self):
        return "{},{},{},{},{},{}".format(self.state,self.action,self.reward,self.nextState,self.nextActions)