'''
Created on Nov 30, 2021

@author: Shufang
'''

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
        
    def getId(self):
        return self.id
    
    def getName(self):
        return self.name
    
    def println(self,time,info):
        print(str(time)+":"+info)