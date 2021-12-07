'''
Created on Nov 30, 2021

@author: Shufang
'''

import com.tao.py.utilities.Log as Log

class Entity(object):
    '''
    classdocs
    '''


    def __init__(self, uuid, name):
        '''
        Constructor
        '''
        self.id=uuid
        self.name=name
        self.index=0
        self.tag=self.__class__.__name__
        
    def getId(self):
        return self.id
    

    def getName(self):
        return self.name
    
    def getIndex(self):
        return self.index
    def setIndex(self,idx): 
        self.index=idx   
    
        
    def i(self,info):
        Log.i(self.tag,info)
    
    def si(self,info,time):
        Log.si(self.tag,info,time)
    
    def d(self,info):
        Log.d(self.tag,info)
    def sd(self,info,time):
        Log.sd(self.tag,info,time) 
        
    def w(self,info):
        Log.w(self.tag,info)
    def sw(self,info,time):
        Log.sw(self.tag,info,time) 
        
    def e(self,info):
        Log.e(self.tag,info)
    def se(self,info,time):
        Log.se(self.tag,info,time)     