from com.tao.py.sim.kernel.SimConfig import SimConfig
from com.tao.py.sim.experiment.Scenario import Scenario
import com.tao.py.utilities.Log as Log
from com.tao.py.rl.environment.Environment4 import SimEnvironment4
from com.tao.py.rl.agent.Agent8_n_step_saras_agg_reward import Agent8
from com.tao.py.rl.policy.AgentPolicy import AgentPolicy
from com.tao.py.manu import ModelFactory


def createModel():
    return ModelFactory.create1M2PModel()


Log.addFilter("INFO")
simConfig=SimConfig(1,100);

scenario=Scenario(1,"S1",simConfig,createModel)

environment=SimEnvironment4(scenario)
# agent=Agent8(environment)
# environment.policy=AgentPolicy(agent,0.2)
# agent.prepare()
# environment.start()
# agent.learn()