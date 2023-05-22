'''
Created on Dec 17, 2021

@author: cquzh
'''


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.utils import common



class KPIsInEpisode(tf_metric.TFStepMetric):
    
    def __init__(self, name='KPIsOverEpisode',kpiName="CT", prefix='Metrics', batch_size=1,dtype=tf.float32, buffer_size=10):
        name=name+"_"+kpiName
        super(KPIsInEpisode, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.batch_size=batch_size
        self.kpiName=kpiName
        self.kpiValue = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name=kpiName)

    @common.function(autograph=True)
    def call(self, trajectory):
        
        #tf.where(trajectory.is_last(), tf.zeros_like(self._return_accumulator),self._return_accumulator))
        indices=tf.where(trajectory.is_last())
        
        def collect():
            lasts=tf.gather(trajectory.observation, tf.reshape(indices,[self.batch_size]))       
            self.kpiValue.assign(tf.where(self.kpiName=="CT",lasts[:,-2],lasts[:,-1])) 
        def emptyFn():
            pass
        tf.cond( tf.not_equal(tf.size(indices), 0),true_fn=collect,false_fn=emptyFn)
        
        
        
        return trajectory
    
        # # Zero out batch indices where a new episode is starting.
        # self._return_accumulator.assign(
        #     tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
        #              self._return_accumulator))
        #
        # # Update accumulator with received rewards. We are summing over all
        # # non-batch dimensions in case the reward is a vector.
        # self._return_accumulator.assign_add(
        #     tf.reduce_sum(
        #         trajectory.reward, axis=range(1, len(trajectory.reward.shape))))
        #
        # # Add final returns to buffer.
        # last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
        # for indx in last_episode_indices:
        #   self._buffer.add(self._return_accumulator[indx])
        #
        # return trajectory
    
    def result(self):
        return self.kpiValue[tf.size(self.kpiValue)-1]
    
    @common.function
    def reset(self):
        self.kpiValue.assign(tf.zeros([self.batch_size], self.dtype))
        
class NumberOfEpisodes(tf_metric.TFStepMetric):
    """Counts the number of episodes in the environment."""
    
    def __init__(self, name='NumberOfEpisodes', prefix='Metrics', dtype=tf.int64):
        super(NumberOfEpisodes, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.number_episodes = common.create_variable(
            initial_value=0, dtype=self.dtype, shape=(), name='number_episodes')
    @common.function(autograph=True)
    def call(self, trajectory):
        """Increase the number of number_episodes according to trajectory.
        
        It would increase for all trajectory.is_last().
        
        Args:
          trajectory: A tf_agents.trajectory.Trajectory
        
        Returns:
          The arguments, for easy chaining.
        """
        # The __call__ will execute this.
        num_episodes = tf.cast(trajectory.is_boundary(), self.dtype)
        num_episodes = tf.reduce_sum(input_tensor=num_episodes)
        self.number_episodes.assign_add(num_episodes)
        return trajectory
    
    def result(self):
        return tf.identity(self.number_episodes, name=self.name)
    
    @common.function
    def reset(self):
        self.number_episodes.assign(0)