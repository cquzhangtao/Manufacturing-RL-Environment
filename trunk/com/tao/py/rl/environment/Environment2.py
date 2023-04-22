'''
Created on Dec 4, 2021

@author: Shufang
'''
from com.tao.py.rl.environment.Environment0 import SimEnvironment0
from com.tao.py.rl.data.TrainDataItem import TrainDataItem




class SimEnvironment2(SimEnvironment0):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=5):
        self.policy=None
        self.stepCounter=0
        self.envState=0
        self.autoRestart=True
        super().__init__(scenario,resultContainerFn,rewardCalculatorFn=rewardCalculatorFn,name=name,init_runs=init_runs)

    
    def clear(self):
        self.stepCounter=0
        self.envState=0
        super().clear()
    
    def getSimEventListeners(self):
        #self.decisionMaking=DecisionEventListener()
        return super().getSimEventListeners()      
    
    def start(self,simListeners=[]):

        super().start(self.policy,training=True,simListeners=simListeners)
        self.updateCurrentState() 
        self.envState=1
    

    
    def restart(self):
        self.stepCounter+=1
        self.start()
    
    def takeAction(self, actionIdx):

        self.stepCounter+=1
        
        #queueIdx=self.getQueueIdx(actionIdx)
        #self.job=self.queue[queueIdx]
        
        #trainData=TrainDataItem(self.state,self.actions[queueIdx],0,None,None)
        event=self.decisionEventListener.decisionMakingEvent.createDecisionMadeEvent(actionIdx)
        #event=DecisionMadeEvent(self.time,self.tool,self.job,self.queue)
        self.decisionEventListener.decisionMakingEvent.addEvenOnTop(event)

        self.sim.resume()

        self.updateCurrentState()
        #print(self.rep)
        self.reward=self.getRewardForStepByStep()
        self.episodTotalReward+=self.reward
        #print(self.episodTotalReward)
        
        #trainData.reward=self.reward        
        #trainData.nextState=self.state
        #trainData.nextActions=self.actions        
        #print(trainData)
        
        
        
        if self.sim.getState()==3: 
            self.simResult.summarizeReplication(self.scenario.getIndex(), self.rep-1)
            print("LEAR {} {} {},Total Reward:{:.6f}".format(self.name,self.rep,self.simResult.toString(self.scenario.getIndex(), self.rep-1),self.episodTotalReward))
            self.kpi.append(self.simResult.getKPI(self.scenario.getIndex(), self.rep-1))             
            self.allEpisodTotalReward.append(self.episodTotalReward)
            self.envState=2   
            if self.autoRestart:       
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
    
    def getJobByIndex(self,actionIdx):
        return self.queue[actionIdx]
    
    def getQueueIdx(self,actionIdx):
        return actionIdx
       
    def updateCurrentState(self):
        #self.tool=self.decisionMaking.tool;
        #self.queue=self.decisionMaking.queue;
        self.time=self.decisionMaking.time
        self.state=self.decisionMaking.getState()
        self.actions=self.getActionSet()

        
             
            
            