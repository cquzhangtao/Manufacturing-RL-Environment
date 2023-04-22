'''
Created on Dec 8, 2021

@author: Shufang
'''
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
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
        self.layer1=Dense(5,activation='sigmoid')
        self.layer2=Dense(5,activation='sigmoid')
        self.layer3=Dense(actionFeatureNum,activation='relu')
        self.layer4=Dense(5,activation='sigmoid')
        self.layer5=Dense(1,activation='relu')
        
    def call(self, inputs, training=None, mask=None):
        #state=[[row[0:self.stateFeatureNum]] for row in inputs]  
        #action=[[row[self.stateFeatureNum:self.stateFeatureNum+self.actionFeatureNum]] for row in inputs] 
        state=inputs[:,0:self.stateFeatureNum]
        action=inputs[:,self.stateFeatureNum:self.stateFeatureNum+self.actionFeatureNum]
        #y=self.inputLayer(state)
        y=self.layer1(state)
        y=self.layer2(y) 
        y=self.layer3(y)
        y=y*action
        # y=self.layer4(y)
        #qvalue=self.layer5(y)
        
        outputs=tf.reduce_sum(y, axis=1, keepdims=True)
        return outputs
        