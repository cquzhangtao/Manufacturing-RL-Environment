# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License

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

import tf_agents 


Log.addFilter("INFO")

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

KERAS_LSTM_FUSED = 2
def createModel():
    return ModelFactory.create1M2PModel()



@gin.configurable
def train_eval(
    root_dir,
    env_name='CartPole-v0',
    num_iterations=100000,
    train_sequence_length=1,
    # Params for QNetwork
    fc_layer_params=(100,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    # Params for summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple train and eval for DQN."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
    simConfig=SimConfig(1,100);

    scenario=Scenario(1,"S1",simConfig,createModel)
    env=SimEnvironment5(scenario)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(SimEnvironment5(scenario))
    
    
    def observation_and_action_constrain_splitter(observation):
        if isinstance(observation,BoundedTensorSpec):
            return tf_agents.specs.from_spec(env._observation_spec_no_mask),None 
        
        observ=observation
        if len(observation.shape)==0:
            a=0
        if len(observation.shape)>1:
            observ=observation[0]
            if observ.shape[0]<100:
                a=1
        print("wwwwwwwwwwww")
        print(observ.shape)
        print(observ)

        a=observ[0:env.environmentSpec.stateFeatureNum]
        print("ssssssssssss")
        observation=tf.expand_dims(a, axis=0)
        action_mask=tf.expand_dims(observ[env.environmentSpec.stateFeatureNum:], axis=0)
        return observation,action_mask

    if train_sequence_length != 1 and n_step_update != 1:
      raise NotImplementedError(
          'train_eval does not currently support n-step updates with stateful '
          'networks (i.e., RNNs)')

    action_spec = tf_env.action_spec()
    num_actions = action_spec.maximum - action_spec.minimum + 1

    if train_sequence_length > 1:
      q_net = create_recurrent_network(
          input_fc_layer_params,
          lstm_size,
          output_fc_layer_params,
          num_actions)
    else:
      q_net = create_feedforward_network(fc_layer_params, num_actions,env)
      train_sequence_length = n_step_update


    # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
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

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy





    # initial_collect_policy = random_tf_policy.RandomTFPolicy(
    #     tf_env.time_step_spec(), tf_env.action_spec(),observation_and_action_constraint_splitter=observation_and_action_constrain_splitter)
    #
    #
    # results = metric_utils.eager_compute(
    #     eval_metrics,
    #     eval_tf_env,
    #     eval_policy,
    #     num_episodes=num_eval_episodes,
    #     train_step=global_step,
    #     summary_writer=eval_summary_writer,
    #     summary_prefix='Metrics',
    # )
    # if eval_metrics_callback is not None:
    #   eval_metrics_callback(results, global_step.numpy())
    # metric_utils.log_metrics(eval_metrics)

    time_step = tf_env._reset()
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0



    def train_step(time_step,policy_state):
      action_step = collect_policy.action(time_step, policy_state)

      next_time_step = tf_env.step(action_step.action)
      experience=(time_step, action_step, next_time_step) 
      policy_state = action_step.state
      time_step=next_time_step

      return tf_agent.train(experience),time_step,policy_state



    for iter in range(num_iterations):
      print("iteration"+str(iter))
      start_time = time.time()

      
      train_loss,time_step,policy_state = train_step(time_step,policy_state)
      time_acc += time.time() - start_time

      if global_step.numpy() % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(),
                     train_loss.loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])



      if global_step.numpy() % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)
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
    tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)


def create_feedforward_network(fc_layer_units, num_actions,env):
  net= sequential.Sequential(
      [dense(10) ,dense(10)]
      + [logits(num_actions)])
  net.build((env.environmentSpec.actionFeatureNum,))
  print(net.summary())
  return net


def create_recurrent_network(
    input_fc_layer_units,
    lstm_size,
    output_fc_layer_units,
    num_actions):
  rnn_cell = tf.keras.layers.StackedRNNCells(
      [fused_lstm_cell(s) for s in lstm_size])
  return sequential.Sequential(
      [dense(num_units) for num_units in input_fc_layer_units]
      + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
      + [dense(num_units) for num_units in output_fc_layer_units]
      + [logits(num_actions)])


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval("D:\\python\\")


if __name__ == '__main__':
  app.run(main)
