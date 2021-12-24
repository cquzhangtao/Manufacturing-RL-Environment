

import os
import time

from absl import app
from absl import flags
from absl import logging

from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from com.tao.py.rl.tf_agents.reinforce_agent import ReinforceAgent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from com.tao.py.rl.tf_agents.prepareEnv import prepare as prepareEnv

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 500,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_boolean('graph_compute',True,"enable graph computation")

FLAGS = flags.FLAGS

from tensorflow.python.ops.summary_ops_v2 import create_file_writer as create_file_writer,record_if as record_if
import datetime
from com.tao.py.rl.tf_agents.mmetrics import KPIsInEpisode
from tf_agents.environments import parallel_py_environment
from tf_agents.system import system_multiprocessing as multiprocessing



def train_eval(
    root_dir,
    num_iterations=1000,
    actor_fc_layers=(123,32,),
    value_net_fc_layers=(149,56,),
    use_value_network=False,
    use_tf_functions=False,
    # Params for collect
    collect_episodes_per_iteration=1,
    replay_buffer_capacity=2000,
    # Params for train
    learning_rate=1e-3,
    gamma=0.9,
    gradient_clipping=None,
    normalize_returns=True,
    value_estimation_loss_coef=0.2,
    # Params for eval
    num_eval_episodes=1,
    eval_interval=10,
    # Params for checkpoints, summaries, and logging
    log_interval=100,
    summary_interval=1,
    summaries_flush_secs=1,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):

    """A simple train and eval for Reinforce."""
    root_dir = os.path.expanduser(root_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(root_dir, current_time,'train')
    eval_dir = os.path.join(root_dir, current_time,'eval')
    
    train_summary_writer = create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()
    
    eval_summary_writer =create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)


    with record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
        env, evalEvn, mask,envs = prepareEnv()
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(envs,start_serially=True))
        eval_tf_env = tf_py_environment.TFPyEnvironment(evalEvn)
        
        eval_metrics = [
          tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
          tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
          KPIsInEpisode(kpiName="CT"),
          KPIsInEpisode(kpiName="Reward")
        ]
        
    
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            env._observation_spec_no_mask,
            tf_env.action_spec(),
            fc_layer_params=actor_fc_layers)
    
        if use_value_network:
            value_net = value_network.ValueNetwork(
               env._observation_spec_no_mask,
              fc_layer_params=value_net_fc_layers)
    
        global_step = tf.compat.v1.train.get_or_create_global_step()
        tf_agent = ReinforceAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            actor_network=actor_net,
            value_network=value_net if use_value_network else None,
            value_estimation_loss_coef=value_estimation_loss_coef,
            gamma=gamma,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
            normalize_returns=normalize_returns,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
            observation_and_action_constraint_splitter=mask)
    
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)
    
        tf_agent.initialize()
    
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
            KPIsInEpisode(kpiName="CT"),
            KPIsInEpisode(kpiName="Reward")
        ]
    
        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
    
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_episodes_per_iteration)
    
        def train_step():
            experience = replay_buffer.gather_all()
            return tf_agent.train(experience)
    
        if use_tf_functions:
            # To speed up collect use TF function.
            collect_driver.run = common.function(collect_driver.run)
            # To speed up train use TF function.
            tf_agent.train = common.function(tf_agent.train)
            train_step = common.function(train_step)
    
        # Compute evaluation metrics.
        metrics = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            use_function=use_tf_functions
        )
        #TODO(b/126590894): Move this functionality into eager_compute_summaries
        if eval_metrics_callback is not None:
            eval_metrics_callback(metrics, global_step.numpy())
    
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    
        timed_at_step = global_step.numpy()
        time_acc = 0
    
        for _ in range(num_iterations):
            start_time = time.time()
            #print("collect data")
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            #print("collected")
            total_loss = train_step()
            replay_buffer.clear()
            time_acc += time.time() - start_time
            
            global_step_val = global_step.numpy()
            if global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
                steps_per_sec = (global_step_val - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step_val
                time_acc = 0
    
            for train_metric in train_metrics:
                train_metric.tf_summaries(
                  train_step=global_step, step_metrics=train_metrics[:2])
              
            if global_step_val % eval_interval == 0:
                metrics = metric_utils.eager_compute(
                  eval_metrics,
                  eval_tf_env,
                  eval_policy,
                  num_episodes=num_eval_episodes,
                  train_step=global_step,
                  summary_writer=eval_summary_writer,
                  summary_prefix='Metrics',
                  use_function=use_tf_functions
                )
                # TODO(b/126590894): Move this functionality into
                # eager_compute_summaries.
                if eval_metrics_callback is not None:
                    eval_metrics_callback(metrics, global_step_val)
    
    train_summary_writer.close()
    eval_summary_writer.close()


def main(_):
    tf.compat.v1.enable_eager_execution(
        config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    tf.compat.v1.enable_v2_behavior()
    #logging.set_verbosity(logging.INFO)
    tf.config.run_functions_eagerly(not FLAGS.graph_compute)
    if not FLAGS.graph_compute:        
        tf.data.experimental.enable_debug_mode()
    train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations,use_tf_functions=FLAGS.graph_compute)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    multiprocessing.handle_main(main)
