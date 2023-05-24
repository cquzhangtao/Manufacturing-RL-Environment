'''
Created on May 22, 2023

@author: xiesh
'''
import matplotlib.pyplot as plt
import numpy

hlLoss=None
hlLR=None
hlReward=None
hlKPI=None
stopChartUpdate=False

def enableDynamicChart(environment,agent):
    initChart()
    environment.onReplicationDone=onRepDone
    agent.onStep=onStep

def on_close(event):
    global stopChartUpdate
    stopChartUpdate=True

def initChart():
    
    global hlLoss,hlLR,hlKPI,hlReward
    fig=plt.figure(figsize = (18,10))
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.subplot(221)
    plt.axis([0,100,12,17])
    hlKPI,=plt.plot([],[])
    plt.title("KPI over replications")
    plt.xlabel("Replication")
    plt.ylabel("KPI")
    
    plt.subplot(222)
    plt.axis([0,100,165000,170000])
    hlReward,=plt.plot([],[])
    plt.title("Avg Reward over replications")
    plt.xlabel("Replication")
    plt.ylabel("Avg Reward")
    
    plt.subplot(223)
    plt.axis([0,5000,0,10000000])
    hlLoss,=plt.plot([], [])
    plt.title("NN loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    
    plt.subplot(224)
    plt.axis([0,5000,0,1])
    hlLR,=plt.plot([], [])
    plt.title("Learning rate")
    plt.xlabel("Step")
    plt.ylabel("learning rate")
    
    plt.ion()
    plt.show()
    


def onStep(step,actual,target,loss,learningRate):
    if stopChartUpdate:
        return
    hlLoss.set_xdata(numpy.append(hlLoss.get_xdata(), len(hlLoss.get_xdata())))
    hlLoss.set_ydata(numpy.append(hlLoss.get_ydata(), loss.numpy()))
    hlLR.set_xdata(numpy.append(hlLR.get_xdata(), len(hlLR.get_xdata())))
    hlLR.set_ydata(numpy.append(hlLR.get_ydata(), learningRate.numpy()))
    if step % 200 ==0:
        plt.draw()
        plt.pause(0.1)
        
def onStepTabular(step,actual,target,loss,learningRate):
    if stopChartUpdate:
        return
    hlLoss.set_xdata(numpy.append(hlLoss.get_xdata(), len(hlLoss.get_xdata())))
    hlLoss.set_ydata(numpy.append(hlLoss.get_ydata(), loss))
    hlLR.set_xdata(numpy.append(hlLR.get_xdata(), len(hlLR.get_xdata())))
    hlLR.set_ydata(numpy.append(hlLR.get_ydata(), learningRate))
    if step % 200 ==0:
        plt.draw()
        plt.pause(0.1)
    
def onRepDone(rep,kpi,reward):
    if stopChartUpdate:
        return
    hlKPI.set_xdata(numpy.append(hlKPI.get_xdata(), len(hlKPI.get_xdata())))
    hlKPI.set_ydata(numpy.append(hlKPI.get_ydata(), kpi))
    hlReward.set_xdata(numpy.append(hlReward.get_xdata(), len(hlReward.get_xdata())))
    hlReward.set_ydata(numpy.append(hlReward.get_ydata(), reward))
    if rep % 2 ==0:
        plt.draw()
        plt.pause(0.1)


def drawChart(environment,agent): 
    
    plt.figure(1,figsize = (18,10))
    plt.subplot(331)
    plt.scatter(range(len(environment.kpi)),environment.kpi)
    plt.title("KPI over replications")
    plt.xlabel("Replication")
    plt.ylabel("KPI")

    plt.subplot(332)
    plt.scatter(range(len(environment.kpi)),environment.allEpisodTotalReward)
    plt.title("Avg Reward over replications")
    plt.xlabel("Replication")
    plt.ylabel("Avg Reward")

    
    chunks=environment.split(environment.kpi,50)
    plt.subplot(333)
    plt.plot(range(len(chunks)),chunks)
    plt.title("KPI over replications")
    plt.xlabel("Replication x 50")
    plt.ylabel("KPI")

    
    chunks=environment.split(environment.allEpisodTotalReward,50)
    plt.subplot(334)
    plt.plot(range(len(chunks)),chunks)
    plt.title("Avg Reward over replications")
    plt.xlabel("Replication x 50")
    plt.ylabel("Avg Reward")
    
    
    plt.subplot(335)
    plt.scatter(range(len(agent.losses)),agent.losses)
    plt.title("loss over step")
    plt.xlabel("step")
    plt.ylabel("loss")
    
    chunks=environment.split(agent.losses,500)    
    plt.subplot(336)
    plt.plot(range(len(chunks)),chunks)
    plt.title("loss over step")
    plt.xlabel("step x 500")
    plt.ylabel("loss")  
    
    idx=7
    if hasattr(agent, "losses1"):
        plt.subplot(330+idx)
        plt.scatter(range(len(agent.losses1)),agent.losses1)
        plt.title("extra loss over step")
        plt.xlabel("step")
        plt.ylabel("loss")        
        idx+=1
        
        chunks=environment.split(agent.losses1,500)    
        plt.subplot(330+idx)
        plt.plot(range(len(chunks)),chunks)
        plt.title("extra loss over step")
        plt.xlabel("step x 500")
        plt.ylabel("loss") 
        idx+=1     
    if hasattr(agent, "hisLearningRate"): 
        
        plt.subplot(330+idx)
        plt.plot(range(len(agent.hisLearningRate)),agent.hisLearningRate)
        plt.title("learning rate over step")
        plt.xlabel("step")
        plt.ylabel("learning rate") 
        idx+=1 
            
    plt.tight_layout()
    
    
    
    drawMoreChart(environment,agent)
    
    
    plt.show()
    
def drawMoreChart(environment,agent): 
    
    plt.figure(2,figsize = (18,10))
    
    idx=1

    if hasattr(agent, "variables"):
        
        vidx=0
        for vari in agent.variables :
            varis=[list(i) for i in zip(*vari)] 
            
            plt.subplot(330+idx)
            
            for i in range(min(len(varis),100)):
                plt.plot(range(len(varis[i])),varis[i])
            plt.title( agent.variableNames[vidx])
            plt.xlabel("step")
            plt.ylabel(str(len(varis))+"variable")
            idx+=1
            vidx+=1
        
    plt.tight_layout()
