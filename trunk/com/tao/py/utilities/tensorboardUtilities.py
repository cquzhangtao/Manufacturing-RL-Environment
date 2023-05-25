'''
Created on May 24, 2023

@author: xiesh
'''
import os
import datetime
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
import tensorflow as tf 
from tensorflow.python.keras.models import Model
import random

def init(model,agent):
    tf.data.experimental.enable_debug_mode()
    tf.config.run_functions_eagerly(True)
    root_dir="~/rl/"+model+"/"+agent
    root_dir = os.path.expanduser(root_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_dir = os.path.join(root_dir, current_time)
    
    summary_writer = create_file_writer(
        root_dir, flush_millis=120 * 1000)
    summary_writer.set_as_default()
    #agent.drawGraph()

def saveStepInfo(agent,step,loss,grads,inputD,output,target,learningRate):
    agent.hisLearningRate.append(learningRate) 
    
    agent.losses.append(loss)
    
    agent.lossesChunks.append(loss)

    if len(agent.lossesChunks)>500:
        tf.summary.scalar("agent/loss_chunk",tf.reduce_mean(agent.lossesChunks),step=step//500)
        agent.lossesChunks=[]
    
    if step % 1000 ==0:      
        if hasattr(agent, "qValues"):
            tf.summary.histogram("tabular_Q_all",agent.qValues,step=step)
            idxRow=0
            for row in agent.qValues:
                idxCol=0
                tf.summary.histogram("tabular_Q_group/state"+str(idxRow),row,step=step)
                for col in row:
                    tf.summary.scalar("tabular_Q/state"+str(idxRow)+"_action"+str(idxCol),col,step=step)
                    idxCol+=1  
                idxRow+=1
    
    
    if step % random.randint(80,120) ==0:
        tf.summary.scalar("agent/loss",tf.reduce_mean(loss),step=step)
        tf.summary.scalar("agent/learning rate",learningRate,step=step) 
        
        idx=0
        for inD in inputD[0]:
            tf.summary.scalar("input/"+str(idx),tf.reduce_mean(inD),step=step)
           
            idx+=1
            
        idx=0
        for out in output[0]:
            tf.summary.scalar("output/"+str(idx),tf.reduce_mean(out),step=step)
            idx+=1 
        
        #for layer in agent.network.layers:
        #    tf.summary.histogram("output_layer/layer"+layer.name,layer.output,step=step)
        if hasattr(agent, "network"):
            for i in range(0, len(agent.network.layers)):
                tmp_model = Model(inputs=agent.network.layers[0].input, outputs=agent.network.layers[i].output)
                tmp_output = tmp_model.predict(inputD)
                tf.summary.histogram("output_layer/"+agent.network.layers[i].name,tmp_output,step=step)   
        

            idx0=0
            for varis in agent.network.trainable_variables:
                vtype="w"
                if idx0 % 2 !=0:
                    vtype="b"
                tf.summary.histogram("variable_big_group/layer"+str(idx0//2)+"_"+vtype,varis,step=step)
                idx1=0
                for var in varis:
                    
                    if len(var.shape)>0:
                        tf.summary.histogram("variable_small_group/layer"+str(idx0//2)+"_"+vtype+"_"+str(idx1),var,step=step)
                        idx2=0
                        for item in var:
                        
                            tf.summary.scalar("grad_var/"+str(idx0//2)+"_"+vtype+"_"+str(idx1)+"_"+str(idx2)+"/v",item,step=step)  
                            idx2+=1
                    else:
                        tf.summary.scalar("grad_var/"+str(idx0//2)+"_"+vtype+"_"+str(idx1)+"/v",var,step=step) 
                    idx1+=1  
                idx0+=1   
                
            idx0=0
            for varis in grads:
                vtype="w"
                if idx0 % 2 !=0:
                    vtype="b"
                tf.summary.histogram("gradient_big_group/layer"+str(idx0//2)+"_"+vtype,varis,step=step)
                idx1=0
                for var in varis:
                    
                    if len(var.shape)>0:
                        tf.summary.histogram("gradient_small_group/layer"+str(idx0//2)+"_"+vtype+"_"+str(idx1),var,step=step)
                        idx2=0
                        for item in var:
                        
                            tf.summary.scalar("grad_var/"+str(idx0//2)+"_"+vtype+"_"+str(idx1)+"_"+str(idx2)+"/g",item,step=step)  
                            idx2+=1
                    else:
                        tf.summary.scalar("grad_var/"+str(idx0//2)+"_"+vtype+"_"+str(idx1)+"/g",var,step=step) 
                    idx1+=1  
                idx0+=1   

    if hasattr(agent, "network"):    
        allVars=[varis.numpy().flatten() for varis in agent.network.trainable_variables]        
        idx=0
        for vari in allVars:
            agent.variables[idx].append(vari)
            idx+=1
            
        if agent.variableNames is None:
            agent.variableNames= [varis.name for varis in agent.network.trainable_variables]