'''
Created on Dec 17, 2021

@author: cquzh
'''


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.utils import common



class KPIsInEpisode(tf_metric.TFStepMetric):
    
    def __init__(self, name='KPIsOverEpisode',kpiName="CT", prefix='Metrics', dtype=tf.float32):
        name=name+"_"+kpiName
        super(KPIsInEpisode, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.kpiName=kpiName
        self.kpiValue = common.create_variable(
            initial_value=0, dtype=dtype, shape=(), name=kpiName)

    
    def call(self, trajectory):

        self.kpiValue.assign(tf.where(self.kpiName=="CT",tf.reduce_mean(trajectory.observation[:,-2]),tf.reduce_mean(trajectory.observation[:,-1]))) 

        return trajectory
    
    def result(self):
        return tf.identity(self.kpiValue, name=self.name)
    
    @common.function
    def reset(self):
        self.kpiValue.assign(0)
        
class NumberOfEpisodes(tf_metric.TFStepMetric):
    """Counts the number of episodes in the environment."""
    
    def __init__(self, name='NumberOfEpisodes', prefix='Metrics', dtype=tf.int64):
        super(NumberOfEpisodes, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.number_episodes = common.create_variable(
            initial_value=0, dtype=self.dtype, shape=(), name='number_episodes')
    
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