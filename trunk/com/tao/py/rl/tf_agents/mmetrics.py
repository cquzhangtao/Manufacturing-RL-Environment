'''
Created on Dec 17, 2021

@author: cquzh
'''


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.utils import common


class KPIsInEpisode(tf_metric.TFStepMetric):
    
    def __init__(self, environment,name='KPIsOverEpisode',kpiName="CT", prefix='Metrics', dtype=tf.float32):
        name+="_"+kpiName
        super(KPIsInEpisode, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.episodeIdx=0
        self.kpiName=kpiName

        self.kpis_episodes = 0
        self.env=environment
    
    def __call__(self, trajectory):
        def append():
            if self.kpiName=="CT":
                self.kpis_episodes=self.env.kpi[self.episodeIdx]
            elif self.kpiName=="Reward":
                self.kpis_episodes=self.env.rewards[self.episodeIdx]
            else:
                self.kpis_episodes=0
            self.episodeIdx+=1
        def donothing():
            pass
         
        tf.cond(trajectory.is_last(),true_fn=append,false_fn=donothing)
        return trajectory
    
    def result(self):
        return tf.identity(self.kpis_episodes, name=self.name)
    
    @common.function
    def reset(self):
        self.kpis_episodes=0
