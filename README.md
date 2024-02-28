# Introduction
In many cases involving Reinforcement Learning (RL), adequately defining and quantifying the state, action, and reward within a simulation or environment poses a significant challenge, often overlooked in RL studies. This project creates a wrapper on simulation and handles the definitions. Eight types of environments are built based on the definition of state, action, and reward function. For example, one environment supports n-step reward, and one supports dynamic and stochastic action sets. Some convert featured action space to categorical action space. Some also implement environment interfaces from the gym and TF-agent. 
# How to use?
## Simulation model
1) create your simulation model which extends the class SimModel.
2) all entities in your model must implement the interface SimEntity.
3) in your model, once a decision needs to be made in an entity, add the following code:
   ```
   event=DecisionMakingEvent(time,self)
   self.addEvent(event)
   ```
4) the decision-making event must implement method decisionMakingEvent.createDecisionMadeEvent() to take the action made by agents.
## Agents
```
def learn(self):
    for _ in range(self.epochs):
        while not self.environment.finishedEpisode():
            trainData=self.environment.collectOneStepData()
            reward=trainData.reward
            # write algorithm here to update Q function.
        self.environment.restart()
```
