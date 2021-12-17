'''
Created on Dec 17, 2021

@author: cquzh
'''


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts


class KPIsInEpisode(tf_metric.TFStepMetric):
    
    def __init__(self, name='KPIsOverEpisode',kpiName="CT", prefix='Metrics', dtype=tf.float32):
        name=name+"_"+kpiName
        super(KPIsInEpisode, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.kpiName=kpiName
        self.kpiValue = common.create_variable(
            initial_value=0, dtype=dtype, shape=(), name=kpiName)

    
    def call(self, trajectory):

        self.kpiValue.assign(tf.where(self.kpiName=="CT",trajectory.observation[0][-2],trajectory.observation[0][-1])) 

        return trajectory
    
    def result(self):
        return tf.identity(self.kpiValue, name=self.name)
    
    @common.function
    def reset(self):
        self.kpiValue.assign(0)
