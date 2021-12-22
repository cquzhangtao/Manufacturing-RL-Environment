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

"""Policy implementation that generates random actions."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function


import tensorflow as tf
import numpy as np

from tf_agents.policies import tf_policy

from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types




# TODO(b/161005095): Refactor into RandomTFPolicy and RandomBanditTFPolicy.
class RandomQtoptPolicy(tf_policy.TFPolicy):
    """Returns random samples of the given action_spec.
    
    Note: the values in the info_spec (except for the log_probability) are random
      values that have nothing to do with the emitted actions.
    
    Note: The returned info.log_probabiliy will be an object matching the
    structure of action_spec, where each value is a tensor of size [batch_size].
    """

    def __init__(self,time_step_spec,action_sepc,max_action_num,*args, **kwargs):
        
        #observation_and_action_constraint_splitter = (kwargs.get('observation_and_action_constraint_splitter', None))
        # self._accepts_per_arm_features = (
        #      kwargs.pop('accepts_per_arm_features', False))
        super(RandomQtoptPolicy, self).__init__(time_step_spec, action_sepc, *args,**kwargs)
        self.max_action_num=max_action_num                                  
        #self.observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
        
    def _variables(self):
        return []
    
    def _action(self, time_step, policy_state, seed):

        actions,actionNum,actionFeatureNum,batchSize=self.getActionsFromObservation(time_step.observation)
        
        index = tf.constant(0)
        selActions=tf.zeros([0,actionFeatureNum])
        condition = lambda index,_: tf.less(index, batchSize)
        #body=lambda index: (tf.add(index, 1),)
        # def body_fn():
        def body(index,selActions):
        
            action=actions[index,tf.random.uniform((), maxval=actionNum[...,index], dtype=tf.dtypes.int32,seed=seed),:]
            action=tf.expand_dims(action,axis=0)
            selActions=tf.concat([selActions,action],axis=0)
            index=tf.add(index, 1)
            return index,selActions
            
        [index,selActions] = tf.while_loop(condition, body, [index,selActions])
        
        #action_=actions[...,tf.random.uniform((), maxval=actionNum, dtype=tf.dtypes.int32,seed=seed),:]
        #action_=tf.expand_dims(action_,axis=0)
        
        step = policy_step.PolicyStep(selActions, policy_state, ())
        return step
    
    def getActionsFromObservation(self, observ):
        net_observation=observ
        observation_and_action_constraint_splitter=(self.observation_and_action_constraint_splitter)
        if observation_and_action_constraint_splitter:
            _,actions=observation_and_action_constraint_splitter(net_observation)
            batch_size=len(observ)
            actionNum=tf.cast(actions[...,0],tf.dtypes.int32)
            actionFeatureNum=tf.cast(actions[...,1],tf.dtypes.int32)
            actionFeatureNum=self._action_spec.shape[0]
            

            
            npActions = actions[...,2:]
            npActions=tf.reshape(npActions,(batch_size,self.max_action_num,actionFeatureNum))
            return npActions,actionNum,actionFeatureNum,batch_size
        
        return None,None
    
    def _distribution(self, time_step, policy_state):
        raise NotImplementedError(
            'RandomTFPolicy does not support distributions yet.')
