'''
Created on Nov 30, 2021

@author: cquzh
'''
from com.tao.py.utilities.Entity import Entity
import copy
from com.tao.py.sim.kernel.Simulator import Simulator

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
                
                model=copy.deepcopy(scenario.getModel())
                model.setReplication(rep) 
                model.setScenario(scenario)              
                sim=Simulator(simConfig,self.eventListeners)
                
                model.setEngine(sim)
                
                for simEntity in model.getSimEntities():
                    simEntity.setEngine(sim)
                    simEntity.setReplication(rep) 
                    simEntity.setScenario(scenario)
                
                sim.run(model)
                
            