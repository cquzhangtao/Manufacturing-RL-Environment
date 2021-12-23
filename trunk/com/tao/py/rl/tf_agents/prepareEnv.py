'''
Created on Dec 16, 2021

@author: cquzh
'''

from com.tao.py.rl.environment.Environment5 import SimEnvironment5
from com.tao.py.rl.environment.Environment6 import SimEnvironment6
from com.tao.py.sim.kernel.SimConfig import SimConfig
from com.tao.py.sim.experiment.Scenario import Scenario
from com.tao.py.manu import ModelFactory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec

import com.tao.py.utilities.Log as Log
import tf_agents
import logging
import functools

logging.disable(logging.WARNING)

Log.addFilter("INFO")


def createModel():
    return ModelFactory.create1M2PModel()

def createEnv(name,scenario):

    return SimEnvironment5(scenario,name=name)

def prepare(num_parallel_environments=1):
    simConfig=SimConfig(1,100);
    
    scenario=Scenario(1,"S1",simConfig,createModel)
    
    envs=[]
    for i in range(num_parallel_environments):
        fun=functools.partial(createEnv,"Train"+str(i),scenario)
        envs.append(fun)
    
    evalEnv=SimEnvironment5(scenario,name="Evaluation")
    
    env=evalEnv
    def observation_and_action_constrain_splitter(observation):
        if isinstance(observation,BoundedTensorSpec):
            return tf_agents.specs.from_spec(env._observation_spec_no_mask),None 
        
        observ=observation
        
        kpiStartIdx=env.environmentSpec.stateFeatureNum+env.actionNum

        if len(observation.shape)==2:
            observ=observation[:,0:env.environmentSpec.stateFeatureNum]

            action_mask=observation[:,env.environmentSpec.stateFeatureNum:kpiStartIdx]
            return observ,action_mask
        
        if len(observation.shape)==3:
            observ=observation[:,:,0:env.environmentSpec.stateFeatureNum]

            action_mask=observation[:,:,env.environmentSpec.stateFeatureNum:kpiStartIdx]
            return observ,action_mask
        
        if len(observation.shape)>3:
            print("ERRRRRRROR")
    if num_parallel_environments==1:
        return env,evalEnv,observation_and_action_constrain_splitter
    if num_parallel_environments>1:
        return env,evalEnv,observation_and_action_constrain_splitter,envs

def createEnv2(name,scenario,observe_spec,observation_spec_no_mask,action_spec):
    env=SimEnvironment6(scenario,name=name)
    env._action_spec=action_spec
    env._observation_spec=observe_spec
    env._observation_spec_no_mask=observation_spec_no_mask
    return env
    
def prepare2(num_parallel_environments=1):
    simConfig=SimConfig(1,100);
    
    scenario=Scenario(1,"S1",simConfig,createModel)
    evalEnv=SimEnvironment6(scenario,name="Evaluation")
    envs=[]
    for i in range(num_parallel_environments):
        fun=functools.partial(createEnv2,"Train"+str(i),scenario,evalEnv._observation_spec,evalEnv._observation_spec_no_mask,evalEnv._action_spec)
        envs.append(fun)
    

    
    env=evalEnv
    def observation_and_action_constrain_splitter(observation):
        if isinstance(observation,BoundedTensorSpec):
            return tf_agents.specs.from_spec(env._observation_spec_no_mask),None 
        
        observ=observation
        
        kpiStartIdx=env.environmentSpec.stateFeatureNum+2+env.maxActionNum*env.environmentSpec.actionFeatureNum

        if len(observation.shape)==2:
            observ=observation[:,0:env.environmentSpec.stateFeatureNum]

            actions=observation[:,env.environmentSpec.stateFeatureNum:kpiStartIdx]
            return observ,actions
        
        if len(observation.shape)==3:
            observ=observation[:,:,0:env.environmentSpec.stateFeatureNum]

            actions=observation[:,:,env.environmentSpec.stateFeatureNum:kpiStartIdx]
            return observ,actions
        
        if len(observation.shape)>3:
            print("ERRRRRRROR")

    return env,evalEnv,observation_and_action_constrain_splitter,envs