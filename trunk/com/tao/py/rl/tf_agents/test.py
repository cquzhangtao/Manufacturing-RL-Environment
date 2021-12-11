'''
Created on Dec 11, 2021

@author: Shufang
'''

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

from com.tao.py.rl.tf_agents import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from com.tao.py.rl.environment.Environment5 import SimEnvironment5
from com.tao.py.sim.kernel.SimConfig import SimConfig
from com.tao.py.sim.experiment.Scenario import Scenario
from com.tao.py.manu import ModelFactory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec

from keras.layers import InputLayer
import com.tao.py.utilities.Log as Log
import tf_agents.trajectories.trajectory as trajectory

Log.addFilter("INFO")

def createModel():
    return ModelFactory.create1M2PModel()

def observation_and_action_constrain_splitter(observation):
    if isinstance(observation,BoundedTensorSpec):
        return observation,None 
    observ=observation
    observation=observ[0:env.environmentSpec.stateFeatureNum]
    action_mask=observ[env.environmentSpec.stateFeatureNum:]
    return observation,action_mask

simConfig=SimConfig(1,100);

scenario=Scenario(1,"S1",simConfig,createModel)
env=SimEnvironment5(scenario)
tf_env = tf_py_environment.TFPyEnvironment(env)
eval_tf_env = tf_py_environment.TFPyEnvironment(SimEnvironment5(scenario))

initial_collect_policy = random_tf_policy.RandomTFPolicy(
    tf_env.time_step_spec(), tf_env.action_spec(),observation_and_action_constraint_splitter=observation_and_action_constrain_splitter)

_trajectory_spec = trajectory.Trajectory(
        step_type=tf_env.time_step_spec().step_type,
        observation=tf_env.time_step_spec().observation,
        action=tf_env.action_spec(),
        policy_info=initial_collect_policy.info_spec,
        next_step_type=tf_env.time_step_spec().step_type,
        reward=tf_env.time_step_spec().reward,
        discount=tf_env.time_step_spec().discount)


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=_trajectory_spec,
    batch_size=tf_env.batch_size,
    max_length=100)


dynamic_step_driver.DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1000).run()
