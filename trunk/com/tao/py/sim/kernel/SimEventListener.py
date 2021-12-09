'''
Created on Nov 30, 2021

@author: Shufang
'''

class SimEventListener(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def onEventTriggered(self,event): 
        pass 
    
    def extend2DArray(self,array,d1,d2):  
        i= len(array)
        while i<=d1:
            i+=1
            array.append([])
        self.extend1DArray(array[d1],d2 )  
    
    def extend1DArray(self,array,d1):
        i= len(array)
        while i<=d1:
            i+=1
            array.append(None)        