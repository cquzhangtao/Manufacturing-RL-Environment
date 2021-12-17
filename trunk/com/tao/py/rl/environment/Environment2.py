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
        self.envState=0
        super().__init__(scenario)

    
    def clear(self):
        self.stepCounter=0
        self.envState=0
        super().clear()
    
    
    def start(self,training=True,rule=None):
        self.decisionMaking=DecisionEventListener()
        self.eventListeners.append(self.decisionMaking)
        if rule==None and self.policy!=None:
            rule=AgentRule(self.policy)
        super().start(training=training,rule=rule)
        self.updateCurrentState()
        self.rewards.append(0) 
        self.envState=1
    
    def getJobByIndex(self,actionIdx):
        return self.queue[actionIdx]
    
    def getQueueIdx(self,actionIdx):
        return actionIdx
    
    def restart(self):
        self.start()
    
    def takeAction(self, actionIdx):

        self.stepCounter+=1
        
        queueIdx=self.getQueueIdx(actionIdx)
        self.job=self.queue[queueIdx]
        
        trainData=TrainDataItem(self.state,self.actions[queueIdx],0,None,None)
        
        event=DecisionMadeEvent(self.time,self.tool,self.job,self.queue)
        self.tool.addEventOnTop(event)

        self.sim.resume()

        self.updateCurrentState()
        #print(self.rep)
        self.reward=self.getRewardForStepByStep()
        self.rewards[self.rep-1]+=self.reward
        
        trainData.reward=self.reward        
        trainData.nextState=self.state
        trainData.nextActions=self.actions        
        #print(trainData)
        
        
        
        if self.sim.getState()==3: 
            print(str(self.rep)+" "+self.simResult.getTotalSummary().toString()+",Total Reward:"+str(self.rewards[self.rep-1]))
            self.kpi.append(self.simResult.getTotalSummary().getAvgCT())  
            self.envState=2          
            self.restart()

    def finishedEpisode(self):
        return  self.envState==2 
    
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

        
             
            
            