'''
Created on Nov 29, 2021

@author: Shufang
'''

class Simulator(object):
    '''
    classdocs
    '''


    def __init__(self, simConfig):
        '''
        Constructor
        '''
        self.simConfig=simConfig
        self.eventList=[]
        self.time=0
        
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
            
    
    def run(self,model):
        self.insertEvents(model.getInitialEvents())
        while len(self.eventList)>0 and self.time<self.simConfig.getSimLen():
            events=self.getFirstEvents()
            self.time=events[0].getTime()
            #print("Time:"+str(self.time))
            for event in events:
                event.trigger()

            
            
        
        