'''
Created on Dec 8, 2021

@author: Shufang
'''
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import InputLayer
import tensorflow as tf

class Network(Model):
    '''
    classdocs
    '''


    def __init__(self, stateFeatureNum,actionFeatureNum,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.stateFeatureNum=stateFeatureNum
        self.actionFeatureNum=actionFeatureNum
        #self.inputLayer=InputLayer(input_shape=(stateFeatureNum,))
        self.layer1=Dense(5)
        self.layer2=Dense(5)
        self.layer3=Dense(actionFeatureNum)
        
    def call(self, inputs, training=None, mask=None):
        #state=[[row[0:self.stateFeatureNum]] for row in inputs]  
        #action=[[row[self.stateFeatureNum:self.stateFeatureNum+self.actionFeatureNum]] for row in inputs] 
        state=inputs[:,0:self.stateFeatureNum]
        action=inputs[:,self.stateFeatureNum:self.stateFeatureNum+self.actionFeatureNum]
        #y=self.inputLayer(state)
        y=self.layer1(state)
        y=self.layer2(y) 
        y=self.layer3(y)
        qvalue=y*action
        outputs=tf.reduce_sum(qvalue, axis=1, keepdims=True)
        return outputs
        