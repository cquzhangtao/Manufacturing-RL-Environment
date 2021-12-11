

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time

from absl import app

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from com.tao.py.rl.tf_agents import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.networks import sequential
from tf_agents.utils import common
from com.tao.py.rl.environment.Environment5 import SimEnvironment5
from com.tao.py.sim.kernel.SimConfig import SimConfig
from com.tao.py.sim.experiment.Scenario import Scenario
from com.tao.py.manu import ModelFactory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec

import com.tao.py.utilities.Log as Log

import tf_agents 
import logging

logging.disable(logging.WARNING)
Log.addFilter("INFO")


def createModel():
    return ModelFactory.create1M2PModel()



@gin.configurable
def train_eval(
    num_iterations=100000,
    epsilon_greedy=0.1,
    target_update_tau=0.05,
    target_update_period=5,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    debug_summaries=False,
    summarize_grads_and_vars=False):


    global_step = tf.compat.v1.train.get_or_create_global_step()

    simConfig=SimConfig(1,100);
    
    scenario=Scenario(1,"S1",simConfig,createModel)
    env=SimEnvironment5(scenario)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    
    
    def observation_and_action_constrain_splitter(observation):
        if isinstance(observation,BoundedTensorSpec):
            return tf_agents.specs.from_spec(env._observation_spec_no_mask),None 
        
        observ=observation

        if len(observation.shape)>1:
            observ=observation[0]

        a=observ[0:env.environmentSpec.stateFeatureNum]

        observation=tf.expand_dims(a, axis=0)
        action_mask=tf.expand_dims(observ[env.environmentSpec.stateFeatureNum:], axis=0)
        return observation,action_mask


    action_spec = tf_env.action_spec()
    num_actions = action_spec.maximum - action_spec.minimum + 1


    q_net = create_feedforward_network(num_actions,env)

    tf_agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        observation_and_action_constraint_splitter=observation_and_action_constrain_splitter)
    tf_agent.initialize()


    collect_policy = tf_agent.collect_policy


    time_step = tf_env._reset()
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    time_acc = 0

    for itera in range(num_iterations):
        #print("iteration"+str(itera))
        start_time = time.time()
        
        action_step = collect_policy.action(time_step, policy_state)
        
        next_time_step = tf_env.step(action_step.action)
        experience=(time_step, action_step, next_time_step) 
        policy_state = action_step.state
        time_step=next_time_step
        train_loss =  tf_agent.train(experience)
        time_acc += time.time() - start_time
     
    return train_loss


logits = functools.partial(
    tf.keras.layers.Dense,
    activation=None,
    kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
    bias_initializer=tf.constant_initializer(-0.2))


dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal'))


fused_lstm_cell = functools.partial(
    tf.keras.layers.LSTMCell, implementation=2)


def create_feedforward_network( num_actions,env):
    net= sequential.Sequential(
        [dense(100) ,dense(100)]
        + [logits(num_actions)])
    net.build((env.environmentSpec.actionFeatureNum,))
    print(net.summary())
    return net


def main(_):
    tf.compat.v1.enable_v2_behavior()
    train_eval()


if __name__ == '__main__':
    app.run(main)