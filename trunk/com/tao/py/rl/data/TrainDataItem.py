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
        
    def getState(self):
        return self.state
    
    def getAction(self):
        return self.action
    
    def __str__(self):
        return "S[{}]====a[{}]====r[{}]====S'[{}]====A'[{}]".format(self.state.__str__(),self.action.__str__(),self.reward,self.nextState.__str__(),";".join(["a{}[{}]".format(idx,action.__str__()) for idx,action in enumerate(self.nextActions)]))
    
    def flatten(self):
        data=[]
        data.extend(self.state.getData())
        data.extend(self.action.getData())
        data.append(self.reward)
        data.extend(self.nextState.getData())
        data.extend([x for action in self.nextActions for x in action.getData()] )
        return data