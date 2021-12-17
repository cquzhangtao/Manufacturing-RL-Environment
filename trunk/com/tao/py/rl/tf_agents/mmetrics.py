'''
Created on Dec 17, 2021

@author: cquzh
'''


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.utils import common


class KPIsInEpisode(tf_metric.TFStepMetric):
    
    def __init__(self, environment,name='KPIsInEpisode', prefix='Metrics', dtype=tf.float32):
        super(KPIsInEpisode, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.episodeIdx=0

        self.kpis_episodes = []
        self.env=environment
    
    def call(self, trajectory):
        def append():
            self.kpis_episodes.append([self.env.kpi[self.episodeIdx],self.env.rewards[self.episodeIdx]])   
            self.episodeIdx+=1
        def donothing():
            pass
         
        tf.cond(trajectory.is_last(),true_fn=append,false_fn=donothing)
        return trajectory
    
    def result(self):
        return tf.identity(self.kpis_episodes, name=self.name)
    
    @common.function
    def reset(self):
        self.kpis_episodes=[]
