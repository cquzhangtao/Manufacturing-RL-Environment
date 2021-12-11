'''
Created on Dec 4, 2021

@author: Shufang
'''

from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent
from com.tao.py.rl.environment.DecisionEventListener import DecisionEventListener

from com.tao.py.manu.rule.Rule import AgentRule
from com.tao.py.rl.environment.Environment0 import SimEnvironment0
from com.tao.py.rl.data.TrainDataItem import TrainDataItem


class SimEnvironment2(SimEnvironment0):

    def __init__(self,scenario):
        self.policy=None
        self.stepCounter=0
        super().__init__(scenario)

    
    def clear(self):
        self.stepCounter=0
        super().clear()
    
    
    def start(self,training=True,rule=None):
        self.decisionMaking=DecisionEventListener()
        self.eventListeners.append(self.decisionMaking)
        if rule==None and self.policy!=None:
            rule=AgentRule(self.policy)
        super().start(training=training,rule=rule)
        self.updateCurrentState()
    
    def getJobByIndex(self,actionIdx):
        return self.queue[actionIdx]
    
    def restart(self):
        self.start()
    
    def takeAction(self, actionIdx):
        self.stepCounter+=1
        job=self.getJobByIndex(actionIdx)
        event=DecisionMadeEvent(self.time,self.tool,job,self.queue)
        self.tool.addEventOnTop(event)

        self.sim.resume()
        
        self.updateCurrentState()
        #print(self.rep)
        self.reward=1/self.simResult.getTotalSummary().getAvgCT()
        
        if self.sim.getState()==3: 
            print(self.simResult.getTotalSummary().toString())
            self.kpi.append(self.simResult.getTotalSummary().getAvgCT())
            self.reset()
            return 0
        return 1    
        
        # if self.sim.getState()==3:  
        #     #return self._reset()      
        #     return ts.termination(np.array([self.state], dtype=np.float32), self.reward)
        # else:
        #     return ts.transition(
        #       np.array([self.state], dtype=np.float32), reward=self.reward, discount=1.0)
    
    
    def collectOneStepData(self): 

        queueIdx,actionIdx=self.policy.getAction(self.state,self.actions)
        self.action=actionIdx
        trainData=TrainDataItem(self.state,self.actions[queueIdx],0,None,None)
        self.takeAction(actionIdx)
        trainData.reward=self.reward
        
        trainData.nextState=self.state
        trainData.nextActions=self.actions
        
        return trainData
    
    
    def updateCurrentState(self):
        self.tool=self.decisionMaking.tool;
        self.queue=self.decisionMaking.queue;
        self.time=self.decisionMaking.time
        self.state=self.getStateFromModel(self.model,self.tool,self.queue,self.time)
        self.actions=self.getActionSetFromQueue(self.queue,self.time)
             
            
            