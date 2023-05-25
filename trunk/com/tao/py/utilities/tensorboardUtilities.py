'''
Created on May 24, 2023

@author: xiesh
'''
import os
import datetime
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
import tensorflow as tf 

def init(model,environment,agent):
    tf.data.experimental.enable_debug_mode()
    tf.config.run_functions_eagerly(True)
    root_dir="~/rl/"+model.getName()+"/"+agent.__class__.__name__
    root_dir = os.path.expanduser(root_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_dir = os.path.join(root_dir, current_time)
    
    summary_writer = create_file_writer(
        root_dir, flush_millis=120 * 1000)
    summary_writer.set_as_default()
    #agent.drawGraph()
