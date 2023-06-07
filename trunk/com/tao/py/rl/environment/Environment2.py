'''
Created on Dec 4, 2021

@author: Shufang
'''
from com.tao.py.rl.environment.Environment0 import SimEnvironment0
from com.tao.py.rl.data.TrainDataItem import TrainDataItem
import tensorflow as tf



class SimEnvironment2(SimEnvironment0):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=100):
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
    
    def start(self,policy,training=True,simListeners=[]):

        super().start(policy,training=training,simListeners=simListeners)
        self.envState=1
        self.updateCurrentState()

    
    def restart(self):
        self.stepCounter+=1
        self.start(self.policy)
    
    def takeAction(self, actionIdx):

        self.stepCounter+=1
        
        idxInActualActionSet=self.getIdxInActualActionSet(actionIdx)
        event=self.decisionEventListener.decisionMakingEvent.createDecisionMadeEvent(idxInActualActionSet)
        self.decisionEventListener.decisionMakingEvent.addEventOnTop(event)

        self.sim.resume()

        self.reward=self.getRewardForStepByStep()
        self.episodTotalReward+=self.reward
       
        self.updateCurrentState()

        if self.sim.getState()==3: 
            self.simResult.summarizeReplication(self.scenario.getIndex(), self.rep-1)
            console="{} {} {},Total Reward:{:.6f}".format(self.name,self.rep,self.simResult.toString(self.scenario.getIndex(), self.rep-1),self.episodTotalReward)
            print(console)
            kpi=self.simResult.getKPI(self.scenario.getIndex(), self.rep-1)
            self.kpi.append(kpi)             
            self.allEpisodTotalReward.append(self.episodTotalReward)


            self.kpiChunk.append(kpi)
            self.totalRewardChunk.append(self.episodTotalReward)
            if self.summaryWriter is None:
                self.summaryWriter=tf.summary
            self.summaryWriter.scalar("env/KPI",kpi,step=self.rep)
            self.summaryWriter.scalar("env/Reward",self.episodTotalReward,step=self.rep) 
            self.summaryWriter.text("env/log", console ,step=self.rep) 
            
            if len(self.kpiChunk)>20:
                self.summaryWriter.scalar("env/KPI_chunk",tf.reduce_mean(self.kpiChunk),step=self.rep)
                self.summaryWriter.scalar("env/Reward_chunk",tf.reduce_mean(self.totalRewardChunk),step=self.rep) 
                self.kpiChunk=[]
                self.totalRewardChunk=[] 
                
            self.envState=2 
            if hasattr(self, "onReplicationDone") and self.onReplicationDone is not None:
                self.onReplicationDone(self.rep,kpi,self.episodTotalReward)   
            if self.autoRestart:       
                self.restart()

    def finishedEpisode(self):
        return  self.envState==2 
    
    def collectOneStepData(self): 

        idxInActualActionSet,idxInFullActionSet=self.policy.getAction(self.getState(),self.getActions())
        #self.action=idxInFullActionSet
        
        trainData=TrainDataItem(self.sim.time,self.getState(),self.getActions()[idxInActualActionSet],0,None,None)

        self.takeAction(idxInFullActionSet)

        trainData.reward=self.reward        
        trainData.nextState=self.getState()
        trainData.nextActions=self.getActions()
        
        if self.summaryWriter is None:
            self.summaryWriter=tf.summary

        idx=0
        for data in trainData.state.getData():
            self.summaryWriter.scalar("env_state/state"+str(idx),data,step=self.stepCounter)
            idx+=1
        idx=0
        for data in trainData.action.getData():
            self.summaryWriter.scalar("env_action/action"+str(idx),data,step=self.stepCounter)
            idx+=1
        self.summaryWriter.scalar("env_reward/imm_reward",trainData.reward,step=self.stepCounter)
        

        
        return trainData
    
    #def getJobByIndex(self,actionIdx):
    #    return self.queue[actionIdx]
    
    def getIdxInActualActionSet(self,actionIdx):
        return actionIdx
       
    def updateCurrentState(self):
        pass

        
             
            
            