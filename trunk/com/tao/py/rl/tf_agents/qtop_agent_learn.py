# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Lint as: python2, python3
r"""Train and Eval DQN.

To run DQN on CartPole:

```bash
tensorboard --logdir $HOME/tmp/dqn/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn/gym/CartPole-v0/ \
  --alsologtostderr
```

To run DQN-RNNs on MaskedCartPole:

```bash
python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn_rnn/gym/MaskedCartPole-v0/ \
  --gin_param='train_eval.env_name="MaskedCartPole-v0"' \
  --gin_param='train_eval.train_sequence_length=10' \
  --alsologtostderr
```

"""

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
from tensorflow.python.ops.summary_ops_v2 import create_file_writer as create_file_writer, record_if as record_if
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from com.tao.py.rl.tf_agents import qtopt_random_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks import nest_map
from tf_agents.keras_layers import inner_reshape

import com.tao.py.rl.tf_agents.qtopt_agent as qtopt_agent
from com.tao.py.rl.tf_agents.prepareEnv import prepare2 as prepareEnv
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.environments import parallel_py_environment
from tf_agents.system import system_multiprocessing as multiprocessing

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_boolean('graph_compute',True,"enable graph computation")
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

import datetime
from com.tao.py.rl.tf_agents.mmetrics import KPIsInEpisode, NumberOfEpisodes


@gin.configurable
def train_eval(
    root_dir,
    num_iterations=10000,
    train_sequence_length=1,

    num_parallel_environments=4,
    critic_obs_fc_layers=(430,),
    critic_action_fc_layers=(10,),
    critic_joint_fc_layers=(300,),

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
    batch_size=7,
    learning_rate=0.001,
    n_step_update=10,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=False,
    # Params for eval
    num_eval_episodes=1,
    eval_interval=500,
    # Params for checkpoints
    train_checkpoint_interval=1000,
    policy_checkpoint_interval=1000,
    rb_checkpoint_interval=1000,
    # Params for summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    eval_metrics_callback=None):
    
    
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(root_dir, current_time,'train')
    eval_dir = os.path.join(root_dir, current_time,'eval')
    
    train_summary_writer = create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()
    
    eval_summary_writer = create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        KPIsInEpisode(kpiName="CT"),
        KPIsInEpisode(kpiName="Reward")
    ]
    
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with record_if(
          lambda: tf.math.equal(global_step % summary_interval, 0)):
          
        env, evalEvn, mask,envs = prepareEnv(num_parallel_environments=num_parallel_environments)
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(envs))
        eval_tf_env = tf_py_environment.TFPyEnvironment(evalEvn)
    
        if train_sequence_length != 1 and n_step_update != 1:
            raise NotImplementedError(
              'train_eval does not currently support n-step updates with stateful '
              'networks (i.e., RNNs)')
    
        
        q_net = create_critic_network(critic_obs_fc_layers,
                                       critic_action_fc_layers,
                                       critic_joint_fc_layers)
        train_sequence_length = n_step_update
        # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
        tf_agent = qtopt_agent.QtOptAgent(
            
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            env.maxActionNum,
            q_net,
            tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            emit_log_probability=False,
            in_graph_bellman_update=True,

            # Params for target network updates
            target_q_network=None,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            enable_td3=False,
            target_q_network_delayed=None,
            target_q_network_delayed_2=None,
            delayed_target_update_period=5,
            # Params for training.
            td_errors_loss_fn=common.element_wise_squared_loss,
            auxiliary_loss_fns=None,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            # Params for debugging
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
            observation_and_action_constraint_splitter=mask,
            )
        tf_agent.initialize()
    
        train_metrics = [
    
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),
            #NumberOfEpisodes(),
            #KPIsInEpisode(kpiName="CT"),
            #KPIsInEpisode(kpiName="Reward")
        ]
    
        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
        initial_collect_policy = qtopt_random_policy.RandomQtoptPolicy(tf_env.time_step_spec(),
            tf_env.action_spec(),env.maxActionNum,observation_and_action_constraint_splitter=mask)

    
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)
    
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=collect_steps_per_iteration)
    
        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=eval_policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)
    
        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()
    
        if use_tf_functions:
            # To speed up collect use common.function.
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)
    
    
        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps with '
            'a random policy.', initial_collect_steps)
        
        
        dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=initial_collect_steps).run()
        print("replay buffer is initialized.")
    
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            use_function=use_tf_functions
        )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)
    
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    
        timed_at_step = global_step.numpy()
        time_acc = 0
    
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)
    
        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)
    
        if use_tf_functions:
            train_step = common.function(train_step)
    
        for _ in range(num_iterations):
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
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
            
            for train_metric in train_metrics[:4]:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])
            for train_metric in train_metrics[4:]:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=[train_metrics[4]])            
            
            # with record_if(time_step.is_last()):
            #     rep = tf.cast(train_metrics[4].result(), tf.int64)
            #     tf.compat.v2.summary.scalar(
            #         name='KPIs/CT', data=time_step.observation[0][-2], step=rep)
            #     tf.compat.v2.summary.scalar(
            #         name='KPIs/Total reward', data=time_step.observation[0][-1], step=rep)
            
            if global_step.numpy() % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step.numpy())
            
            if global_step.numpy() % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step.numpy())
            
            if global_step.numpy() % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step.numpy())
            
            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(
                  eval_metrics,
                  eval_tf_env,
                  eval_policy,
                  num_episodes=num_eval_episodes,
                  train_step=global_step,
                  summary_writer=eval_summary_writer,
                  summary_prefix='Metrics',
                  use_function=use_tf_functions
                  )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(eval_metrics)
        
    #env.drawKPICurve()
    return train_loss


dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=1. / 3.0, mode='fan_in', distribution='uniform'))



def create_identity_layer():
    return tf.keras.layers.Lambda(lambda x: x)

def create_fc_network(layer_units):
    return sequential.Sequential([dense(num_units) for num_units in layer_units])


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


def main(_):
    #logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations,use_tf_functions=FLAGS.graph_compute)


if __name__ == '__main__':

    # with tf.compat.v1.Session() as sess:
    flags.mark_flag_as_required('root_dir')
    multiprocessing.handle_main(functools.partial(app.run, main))
