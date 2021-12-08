'''
Created on Nov 29, 2021

@author: Shufang
'''

class Simulator(object):
    '''
    classdocs
    '''


    def __init__(self, simConfig,eventListeners):
        '''
        Constructor
        '''
        self.simConfig=simConfig
        self.eventList=[]
        self.currentEventList=[]
        self.time=0
        self.eventListeners=eventListeners
        self.state=0 #0 idle 1 busy 2 pause 3 done
        
    def getFirstEvents(self):
        events=[]
        
        
        if len(self.eventList)==1:
            events.append(self.eventList[0])
            del self.eventList[0]
        else:
            index=0
            curEventlen=len(self.eventList) 
            for index in range(curEventlen):
                events.append(self.eventList[index])
                if index==curEventlen-1 or self.eventList[index].getTime()!= self.eventList[index+1].getTime():
                    break           
            del self.eventList[:index+1]
        
            events.sort(key=lambda x: x.priority)
        
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
     
    def insertEventIntoCurrentList(self,newEvent):
        index=0
        for event in self.currentEventList:
            if event.getPriority()<newEvent.getPriority():
                break
            index=index+1
        self.currentEventList.insert(index,newEvent)
        
    def insertEventOnTop(self,newEvent):   
        self.currentEventList.insert(0,newEvent)                   
    
    def run(self):

        while self.state!=2 and self.time<self.simConfig.getSimLen() and self.advance():
            pass

        if self.state!=2:
            self.state=3   
            
    def advance(self):
        if len(self.eventList)+len(self.currentEventList)==0:
            self.state=3  
            return False
        if len(self.currentEventList)==0:
            self.currentEventList=self.getFirstEvents()
        event=self.currentEventList[0]
        del self.currentEventList[0]
        self.time=event.getTime()        
        event.trigger()
        for lis in self.eventListeners:
            lis.onEventTriggered(event)            
            
        return True
        
    def start(self,model):
        model.setEngine(self)
        for simEntity in model.getSimEntities():
            simEntity.setEngine(self)
        model.insertInitialEvents()
        self.state=1
        self.run()

        
    def pause(self): 
        self.state=2 
        
    def resume(self):   
        self.state=1
        self.run()  
    def getState(self):
        return self.state