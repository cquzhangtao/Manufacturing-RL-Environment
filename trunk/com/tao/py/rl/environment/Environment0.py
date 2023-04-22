'''
Created on Dec 4, 2021

@author: Shufang
'''

import pickle
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.rl.data.TrainDataCollectors import TrainDataCollectors
from com.tao.py.rl.data.TrainDataset import TrainDataset
import matplotlib.pyplot as plt
from com.tao.py.rl.event.DecisionEventListener import DecisionEventListener
from com.tao.py.rl.policy.RandomPolicy import RandomPolicy



class SimEnvironment0(object):

    def __init__(self,scenario,resultContainerFn,rewardCalculatorFn=None,name="",init_runs=5):
        self.scenario=scenario
        self.state=None
        self.name=name
        self.rep=0
        self.eventListeners=[]
        self.simResult=resultContainerFn()
        self.kpi=[]
        self.allEpisodTotalReward=[]
        self.episodTotalReward=0
        
        self.environmentSpec=None
        self.initializing=False
        self.rewardCalculator=None
        self.rewardCalculatorFn=rewardCalculatorFn
 

        self.init(repNum=init_runs)
        
        print("Action feature num:{}, State feature num:{}".format(self.environmentSpec.actionFeatureNum,self.environmentSpec.stateFeatureNum))
        print("Max action feature:{}, Max state feature:{}".format(self.environmentSpec.maxAction,self.environmentSpec.maxState))
        print("Min action feature:{}, Min state feature:{}".format(self.environmentSpec.minAction,self.environmentSpec.minState))
        print("Action feature count:{}, State feature count:{}".format(self.environmentSpec.countAction,self.environmentSpec.countState))
    
    def clear(self):
        self.rep=0        
        self.kpi=[] 
        self.allEpisodTotalReward=[]
        self.episodTotalReward=0 

        self.simResult.reset() 

        self.initializing=False;
        
    def getSimEventListeners(self):
        self.decisionEventListener=DecisionEventListener()
        
        if self.rewardCalculatorFn!=None:
            self.rewardCalculator=self.rewardCalculatorFn()       
            return [self.simResult,self.decisionEventListener,self.rewardCalculator]
        else:
            return [self.simResult,self.decisionEventListener]
    
    def start(self,policy,training=False,simListeners=[]):
        self.rep+=1
        #print(self.name+str(self.rep)+"starts")
        self.eventListeners=[]
        self.eventListeners.extend(self.getSimEventListeners())
        self.eventListeners.extend(simListeners)
        
        self.model=self.scenario.createModel()       
        self.model.setReplication(self.rep-1) 
        self.model.setScenario(self.scenario)
        self.model.training=training 
        
        self.episodTotalReward=0
        

        
        self.model.applyPolicy(policy)
    
            
        self.sim=Simulator(self.scenario.getSimConfig(),self.eventListeners)                
                
        for simEntity in self.model.getSimEntities():
            simEntity.setReplication(self.rep-1) 
            simEntity.setScenario(self.scenario)
            simEntity.training=training   

        self.sim.start(self.model)
        
    

            
    def collectData(self,policy,repNum=1): 

        trainDataCollector=TrainDataCollectors(self) 
        
        for _ in range(repNum):
            self.start(policy,training=False,simListeners=[trainDataCollector])
        


        trainDataset=TrainDataset(trainDataCollector)
        
        while len(trainDataset.rawData)==0:
            for _ in range(repNum):
                self.start(policy,training=False,simListeners=[trainDataCollector])
            trainDataset=TrainDataset(trainDataCollector)   
        
        if not self.initializing:
            self.simResult.summarizeReplication(self.scenario.getIndex(),self.rep-1)
            self.kpi.append(self.simResult.getKPI(self.scenario.getIndex(),self.rep-1))
            self.episodTotalReward=sum([j for sub in trainDataset.reward for j in sub])        
            print("{},Total Reward:{:.6f}".format(self.simResult.toString(self.scenario.getIndex(),self.rep-1),self.episodTotalReward))
            self.allEpisodTotalReward.append(self.episodTotalReward)    
        
        return trainDataset
    
    def init(self,repNum=5):
        self.initializing=True
        self.environmentSpec=self.collectData(RandomPolicy(self),repNum=repNum)
        self.clear()
        self.initializing=False
        
    

    
    def getReward(self): 
        #return 10-time+job.getReleaseTime()       
        #return self.simResult.getReplicationSummary(scenario,replication).getAvgCT()
        #return 1/len(queue)
        #return 10-job.getProcessTime()
        if self.rewardCalculator !=None: 
            return self.rewardCalculator.getReward(self)
        return -1
    
    
    def getRewardForStepByStep(self): 
        return self.getReward() 
    
    def saveSpec(self,path): 
        with open(path, 'wb') as outp:
            pickle.dump(self.environmentSpec.max, outp, pickle.HIGHEST_PROTOCOL)
    def loadSpec(self,path):
        self.environmentSpec=TrainDataset(None)
        with open(path, 'rb') as inp:
            self.environmentSpec.max = pickle.load(inp)
    
    def drawKPICurve(self): 
        _, ax1 = plt.subplots()
        ax1.plot(self.kpi)
        plt.title("Avg CT over replications")
        plt.xlabel("Replication")
        plt.ylabel("Avg CT")
        _, ax2 = plt.subplots()
        ax2.plot(self.allEpisodTotalReward)
        plt.show()