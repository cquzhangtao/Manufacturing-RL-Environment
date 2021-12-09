'''
Created on Dec 4, 2021

@author: Shufang
'''

from com.tao.py.manu.event.DecisionMadeEvent import DecisionMadeEvent
from com.tao.py.rl.environment.DecisionEventListener import DecisionEventListener

from com.tao.py.rl.environment.Environment2 import SimEnvironment2
from com.tao.py.rl.data.TrainDataItem import TrainDataItem


class SimEnvironment3(SimEnvironment2):

    def __init__(self,scenario):
        super().__init__(scenario)
    
 
             
            
            