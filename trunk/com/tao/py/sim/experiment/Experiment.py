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


    def __init__(self, uuid,name,scenarios):
        '''
        Constructor
        '''
        super().__init__(uuid,name)
        self.scenarios=scenarios
        
    def start(self):
        for scenario in self.scenarios:
            simConfig=scenario.getSimConfig()
            for rep in range(simConfig.getReplicaiton()):
                
                model=copy.deepcopy(scenario.getModel())
                model.setReplication(rep)               
                sim=Simulator(simConfig)
                
                for simEntity in model.getSimEntities():
                    simEntity.setModel(model)
                    simEntity.setEngine(sim)
                
                sim.run(model)
                
            