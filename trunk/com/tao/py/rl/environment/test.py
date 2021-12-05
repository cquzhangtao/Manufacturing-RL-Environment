'''
Created on Dec 5, 2021

@author: Shufang

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import tf_metrics
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')


FLAGS = flags.FLAGS
from com.tao.py.rl.environment.Environment import CardGameEnv
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent



def train_eval(
    num_iterations=2000000,
    actor_fc_layers=(400, 300),
    critic_obs_fc_layers=(400,),
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    num_parallel_environments=1,
    replay_buffer_capacity=100000,
    ou_stddev=0.2,
    ou_damping=0.15,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    dqda_clipping=None,
    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    gamma=0.995,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for checkpoints, summaries, and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):


    tf_env =tf_py_environment.TFPyEnvironment(CardGameEnv())
  
    
    actor_net = create_actor_network(actor_fc_layers, tf_env.action_spec())
    critic_net = create_critic_network(critic_obs_fc_layers,
                                       critic_action_fc_layers,
                                       critic_joint_fc_layers)
    
    tf_agent =DdpgAgent(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      ou_stddev=ou_stddev,
      ou_damping=ou_damping,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      dqda_clipping=dqda_clipping,
      td_errors_loss_fn=td_errors_loss_fn,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=None)
    tf_agent.initialize()
    
   
    

dense = functools.partial(
  tf.keras.layers.Dense,
  activation=tf.keras.activations.relu,
  kernel_initializer=tf.compat.v1.variance_scaling_initializer(
      scale=1./ 3.0, mode='fan_in', distribution='uniform'))


def create_identity_layer():
    return tf.keras.layers.Lambda(lambda x: x)


def create_fc_network(layer_units):
    return sequential.Sequential([dense(num_units) for num_units in layer_units])


def create_actor_network(fc_layer_units, action_spec):
    """Create an actor network for DDPG."""
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action tensor is supported by this network')
    flat_action_spec = flat_action_spec[0]
    
    fc_layers = [dense(num_units) for num_units in fc_layer_units]
    
    num_actions = flat_action_spec.shape.num_elements()
    action_fc_layer = tf.keras.layers.Dense(
        num_actions,
        activation=tf.keras.activations.tanh,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003))
    
    scaling_layer = tf.keras.layers.Lambda(
        lambda x: common.scale_to_spec(x, flat_action_spec))
    return sequential.Sequential(fc_layers + [action_fc_layer, scaling_layer])
    
    
def create_critic_network(obs_fc_layer_units,
                          action_fc_layer_units,
                          joint_fc_layer_units):
  """Create a critic network for DDPG."""

  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  obs_network = create_fc_network(
      obs_fc_layer_units) if obs_fc_layer_units else create_identity_layer()
  action_network = create_fc_network(
      action_fc_layer_units
  ) if action_fc_layer_units else create_identity_layer()
  joint_network = create_fc_network(
      joint_fc_layer_units) if joint_fc_layer_units else create_identity_layer(
      )
  value_fc_layer = tf.keras.layers.Dense(
      1,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003))

  return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({
          'observation': obs_network,
          'action': action_network
      }),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_fc_layer,
      inner_reshape.InnerReshape([1], [])
  ])





def main():
    tf.compat.v1.enable_v2_behavior()
    train_eval()

if __name__ == '__main__':
#flags.mark_flag_as_required('root_dir')
    main()