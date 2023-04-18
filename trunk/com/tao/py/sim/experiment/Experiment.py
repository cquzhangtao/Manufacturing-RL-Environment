'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity
import copy
from com.tao.py.sim.kernel.Simulator import Simulator
from com.tao.py.manu.rule.Rule import AgentAppRule

class Experiment(Entity):
    '''
    classdocs
    '''


    def __init__(self, uuid,name,scenarios, eventListeners):
        '''
        Constructor
        '''
        super().__init__(uuid,name)
        self.scenarios=scenarios
        self.eventListeners=eventListeners
        
    def start(self):
        for scenario in self.scenarios:
            simConfig=scenario.getSimConfig()
            for rep in range(simConfig.getReplication()):
                
                
                rule=scenario.rule
                if isinstance(scenario.rule,AgentAppRule):
                    
                    agent=Agent8(environment)
                    environment.policy=AgentPolicy(agent,0.2)
                    rule=AgentAppRule()
                
                model=scenario.createModel()
                
                
                    
                
                model.setReplication(rep) 
                model.setScenario(scenario)              
                sim=Simulator(simConfig,self.eventListeners)
                
                
                for simEntity in model.getSimEntities():
                    simEntity.setReplication(rep) 
                    simEntity.setScenario(scenario)
                
                sim.start(model)
                
            