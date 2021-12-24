'''
Created on Dec 4, 2021

@author: Shufang
'''
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.manu.stat.SimDataCollector import SimDataCollector
from com.tao.py.rl.data.TrainDataCollectors import TrainDataCollectors
from com.tao.py.rl.data.TrainDataset import TrainDataset
from com.tao.py.rl.kernel.State import State
from com.tao.py.rl.kernel.Action import Action
from com.tao.py.manu.rule.Rule import AgentRule, FIFORule, RandomRule
import matplotlib.pyplot as plt



class SimEnvironment0(object):

    def __init__(self,scenario,rewardCalculator,name="",init_runs=5):
        self.scenario=scenario
        self.state=None
        self.name=name
        self.rep=0
        self.eventListeners=[]
        self.simResult=None
        self.kpi=[]
        self.allEpisodTotalReward=[]
        self.episodTotalReward=0
        self.rewardCalculator=rewardCalculator
        self.environmentSpec=None
        self.initializing=False;
        self.init(repNum=init_runs)
    
    def clear(self):
        self.rep=0        
        self.kpi=[] 
        self.allEpisodTotalReward=[]
        self.episodTotalReward=0   
        
    def getSimEventListeners(self):
        self.simResult=SimDataCollector()
        return [self.simResult,self.rewardCalculator]
    
    def start(self,training=False,rule=FIFORule(),simListeners=[]):
        self.eventListeners=[]
        self.eventListeners.extend(self.getSimEventListeners())
        self.eventListeners.extend(simListeners)
        
        self.model=self.scenario.createModel()       
        self.model.setReplication(self.rep) 
        self.model.setScenario(self.scenario)
        self.model.training=training 
        
        self.episodTotalReward=0
        
        self.rewardCalculator.reset()
        
        for machine in self.model.machines:
            machine.rule=rule
            
        self.sim=Simulator(self.scenario.getSimConfig(),self.eventListeners)                
                
        for simEntity in self.model.getSimEntities():
            simEntity.setReplication(self.rep) 
            simEntity.setScenario(self.scenario)
            simEntity.training=training   

        self.sim.start(self.model)
        self.rep+=1
    

            
    def collectData(self,policy,rule=None,repNum=1): 

        trainDataCollector=TrainDataCollectors(self) 


        if rule==None:
            rule=AgentRule(policy)
        
        for _ in range(repNum):
            self.start(training=False,simListeners=[trainDataCollector],rule=rule)
        


        trainDataset=TrainDataset(trainDataCollector)
        
        while len(trainDataset.rawData)==0:
            for _ in range(repNum):
                self.start(training=False,rule=rule)
            trainDataset=TrainDataset(trainDataCollector)   
        
        self.kpi.append(self.simResult.getTotalSummary().getAvgCT())
        self.episodTotalReward=sum([j for sub in trainDataset.reward for j in sub])
        print(self.simResult.getTotalSummary().toString()+",Total Reward:"+str(self.episodTotalReward))
        self.allEpisodTotalReward.append(self.episodTotalReward)    
        
        return trainDataset
    
    def init(self,repNum=5):
        self.initializing=True
        self.environmentSpec=self.collectData(None, rule=RandomRule(),repNum=repNum)
        self.clear()
        self.initializing=False
        
    
    def getStateFromModel(self,model,tool,queue,time):
        return State([time,len(queue)])
    
    
    def getActionFromJob(self,job,time): 
        return Action([job.getProcessTime(),time-job.getReleaseTime()])
    
       
    def getActionSetFromQueue(self,queue,time):  
        actions=[]
        for job in queue:
            actions.append(self.getActionFromJob(job,time))  
            
        return actions
    
    def getReward(self,scenario,replication,model,tool,queue,job,time): 
        #return 10-time+job.getReleaseTime()       
        #return self.simResult.getReplicationSummary(scenario,replication).getAvgCT()
        #return 1/len(queue)
        #return 10-job.getProcessTime()
        return self.rewardCalculator.getReward(scenario,replication,model,tool,queue,job,time)
    
    def getRewardForStepByStep(self): 
        return self.getReward(self.scenario.getIndex(), self.rep-1, self.model, self.tool, self.queue, self.job, self.time)        

    
    def drawKPICurve(self): 
        _, ax1 = plt.subplots()
        ax1.plot(self.kpi)
        plt.title("Avg CT over replications")
        plt.xlabel("Replication")
        plt.ylabel("Avg CT")
        _, ax2 = plt.subplots()
        ax2.plot(self.rewards)
        plt.show()