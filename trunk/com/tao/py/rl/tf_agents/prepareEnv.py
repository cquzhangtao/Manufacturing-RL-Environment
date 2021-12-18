'''
Created on Dec 16, 2021

@author: cquzh
'''

from com.tao.py.rl.environment.Environment5 import SimEnvironment5
from com.tao.py.sim.kernel.SimConfig import SimConfig
from com.tao.py.sim.experiment.Scenario import Scenario
from com.tao.py.manu import ModelFactory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec

import com.tao.py.utilities.Log as Log
import tf_agents
import tensorflow as tf
import logging

logging.disable(logging.WARNING)

Log.addFilter("INFO")


def createModel():
    return ModelFactory.create1M2PModel()

def prepare():
    simConfig=SimConfig(1,100);
    
    scenario=Scenario(1,"S1",simConfig,createModel)
    env=SimEnvironment5(scenario,name="Train")
    evalEnv=SimEnvironment5(scenario,name="Evaluation")
    
    
    def observation_and_action_constrain_splitter(observation):
        if isinstance(observation,BoundedTensorSpec):
            return tf_agents.specs.from_spec(env._observation_spec_no_mask),None 
        
        observ=observation
        
        kpiStartIdx=env.environmentSpec.stateFeatureNum+env.actionNum

        if len(observation.shape)==2:
            observ=observation[0]

            a=observ[0:env.environmentSpec.stateFeatureNum]
    
            observation=tf.expand_dims(a, axis=0)
            action_mask=tf.expand_dims(observ[env.environmentSpec.stateFeatureNum:kpiStartIdx], axis=0)
            return observation,action_mask
        
        if len(observation.shape)==3:
            observ=observation[0]

            a=observ[:,0:env.environmentSpec.stateFeatureNum]
    
            observation=tf.expand_dims(a, axis=0)
            action_mask=tf.expand_dims(observ[:,env.environmentSpec.stateFeatureNum:kpiStartIdx], axis=0)
            return observation,action_mask

    return env,evalEnv,observation_and_action_constrain_splitter