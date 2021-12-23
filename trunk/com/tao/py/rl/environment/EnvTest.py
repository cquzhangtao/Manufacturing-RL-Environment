'''
Created on Dec 23, 2021

@author: Shufang
'''
from tf_agents.environments import parallel_py_environment
from com.tao.py.rl.tf_agents.prepareEnv import prepare as prepareEnv
from tf_agents.environments import utils
from tf_agents.system import system_multiprocessing as multiprocessing

def main(arg):
    env, evalEvn, mask,envs = prepareEnv(num_parallel_environments=8)
    environment=parallel_py_environment.ParallelPyEnvironment(envs,start_serially=True)
    utils.validate_py_environment(environment, episodes=16,observation_and_action_constraint_splitter=mask)
if __name__ == '__main__':
    multiprocessing.handle_main(main)
