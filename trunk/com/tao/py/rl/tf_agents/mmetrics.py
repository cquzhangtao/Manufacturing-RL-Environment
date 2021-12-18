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

        self.kpiValue.assign(tf.where(self.kpiName=="CT",trajectory.observation[0][-2],trajectory.observation[0][-1])) 

        return trajectory
    
    def result(self):
        return tf.identity(self.kpiValue, name=self.name)
    
    @common.function
    def reset(self):
        self.kpiValue.assign(0)
        
    def tf_summaries(self, train_step=None, step_metrics=()):
        """Generates summaries against train_step and all step_metrics.
    
        Args:
          train_step: (Optional) Step counter for training iterations. If None, no
            metric is generated against the global step.
          step_metrics: (Optional) Iterable of step metrics to generate summaries
            against.
    
        Returns:
          A list of summaries.
        """
        summaries = []
        prefix = self._prefix
        tag = common.join_scope(prefix, self.name)
        result = self.result()
        
        if train_step is not None:
            summaries.append(
              tf.compat.v2.summary.scalar(name=tag, data=result, step=train_step))
        if prefix:
            prefix += '_'
        for step_metric in step_metrics:
            # Skip plotting the metrics against itself.
            if self.name == step_metric.name:
                continue
            step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
            # Summaries expect the step value to be an int64.
            step = tf.cast(step_metric.result(), tf.int64)
            print(str(step)+str(result))
            summaries.append(tf.compat.v2.summary.scalar(
                name=step_tag,
                data=result,
                step=step))
        return summaries
