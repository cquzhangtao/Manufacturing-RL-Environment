'''
Created on Nov 30, 2021

@author: Shufang
'''

class SimEventListener(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    def onEventTriggered(self,event): 
        pass 
    
    def extendArray(self,array,d1,d2):  
        i= len(array)
        while i<=d1:
            i+=1
            array.append([])
        i=len(array[d1] )  
        while i<=d2:
            i+=1
            array[d1].append(None)        