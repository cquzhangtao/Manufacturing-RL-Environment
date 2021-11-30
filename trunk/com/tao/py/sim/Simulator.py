'''
Created on Nov 29, 2021

@author: Shufang
'''

class Simulator(object):
    '''
    classdocs
    '''


    def __init__(self, simLen,model):
        '''
        Constructor
        '''
        self.simLen=simLen
        self.eventList=[]
        self.insertEvents(model.getInitialEvents())
        self.model=model
        self.time=0
        
    def getFirstEvents(self):
        events=[]
        index=0
        for index in range(len(self.eventList)-1):
            events.append(self.eventList[index])
            if self.eventList[index].getTime()!= self.eventList[index+1].getTime():
                break
 
        if len(self.eventList)==1:
            events.append(self.eventList[0])
            
        del self.eventList[:index+1]
        
        return events    
    
    def insertEvent(self,newEvent):
        index=0
        for event in self.eventList:
            if event.getTime()>newEvent.getTime():
                break
            index=index+1
        self.eventList.insert(index,newEvent)
        
    def insertEvents(self,newEvents):
        for event in newEvents:
            self.insertEvent(event)
    
    def run(self):
        while len(self.eventList)>0 and self.time<self.simLen:
            events=self.getFirstEvents()
            self.time=events[0].getTime()
            #print("Time:"+str(self.time))
            for event in events:
                self.insertEvents(event.trigger())
            
            
        
        