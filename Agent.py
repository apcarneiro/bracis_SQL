import gymnasium as gym
import numpy as np
import random

from SomaticMarker import *

class Agent:

    def setEnv(self, env:gym.Env):
        pass
    def getEnv(self):
        pass

    def setTraningMode(self, learning_rate = 0.7, gamma = 0.95, epsilon = 0.2):
        pass
    
    def setExecutionMode(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def done(self):
        pass

    def getQTable(self):
        pass

    def setQTable(self, qtable):
        pass

    def getStateSize(self):
        pass

    def getActionSize(self):
        pass

    def reset_qtable(self):
        pass



# Agente Q-Learning para o FrozenLake 
class QLearning(Agent):

    def __init__ (self, env:gym.Env):
        self._trainingMode = False
        self._env = env
        self._qtable = np.zeros((env.observation_space.n, env.action_space.n))
        self.reset()    

    def setEnv(self, env:gym.Env):
        self._env = env
        self.reset()

    def getEnv(self):
        return self._env

    def setTraningMode(self, learning_rate = 0.7, gamma = 0.95, epsilon = 0.2):
        self._trainingMode = True
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon

    def setExecutionMode(self):
        self._trainingMode = False

    def step(self):
        if self._trainingMode:
            action = self._epsilon_greedy_policy(self._epsilon)
        else:
            action = self._greedy_policy()

        new_state, reward, done, trunc, _ = self._env.step(action)

        if self._trainingMode:
            #Bellman equation ( Q-learning TD - Temporal-difference)
            self._qtable[self._state,action] = self._qtable[self._state,action] + self._learning_rate * (reward + self._gamma * np.max(self._qtable[new_state,:]) - self._qtable[self._state,action])        
        
        self._done = done or trunc

        old_state = self._state

        self._state = new_state

        return old_state, action, new_state, reward


    def reset(self):
        self._state = self._env.reset()[0]
        self._done = False
 
    def done(self):
        return self._done

    def getQTable(self):
        return self._qtable
    
    def setQTable(self, qtable):
        self._qtable = qtable
    
    def getStateSize(self):
        return self._env.observation_space.n

    def getActionSize(self):
        return self._env.action_space.n
    
    def reset_qtable(self):
        self._qtable = np.zeros((self._env.observation_space.n, self._env.action_space.n))

    
    def _greedy_policy(self):
        max_ids = np.where(self._qtable[self._state, :] == max(self._qtable[self._state, :]))[0]
        action = random.choice(max_ids)
        return action

    def _epsilon_greedy_policy(self, epsilon):
        random_int = random.uniform(0.0,1.0)
        if random_int > epsilon:
            action = self._greedy_policy()
        else:
            action = self._env.action_space.sample()
        return action

# Agente Q-Learning Somático para o FrozenLake 
class SomaticQLearning(Agent):

    def __init__ (self, env:gym.Env, la, ld, d, f:float):
        self._trainingMode = False
        self._env = env
        self._qtable = np.zeros((env.observation_space.n, env.action_space.n))
        
        self._emotions ={
            "frustration": self._frustration,
            "pain": self._pain,
            "euphoria": self._euphoria,
        }

        self._sm = SomaticMarker(la,ld,d,self._emotions)

    def setEnv(self, env:gym.Env):
        self._env = env
        self.reset()

    def getEnv(self):
        return self._env

    def setTraningMode(self, learning_rate = 0.7, gamma = 0.95, epsilon = 0.2):
        self._trainingMode = True
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon

    def setExecutionMode(self):
        self._trainingMode = False

    def step(self):
        if self._trainingMode:
            action = self._epsilon_greedy_policy(self._epsilon)
        else:
            action = self._greedy_policy()

        new_state, reward, done, trunc, _ = self._env.step(action)

        if self._trainingMode:
            #Bellman equation ( Q-learning TD - Temporal-difference)
            self._qtable[self._state,action] = self._qtable[self._state,action] + self._learning_rate * (reward + self._gamma * np.max(self._qtable[new_state,:]) - self._qtable[self._state,action])        
        
        self._done = done or trunc

        old_state = self._state

        ######### SOMATIC LEARNING 
        # sinalize to somatic marker if fall in hole
        if done and reward == 0:
            rw = -1
        else:
            rw = reward
        
        self._sm.learning(old_state,action,new_state,rw)
        #################################################
        
        
        self._state = new_state

        return old_state, action, new_state, reward


    def reset(self):
        self._state = self._env.reset()[0]
        self._done = False
 
    def done(self):
        return self._done

    def getQTable(self):
        return self._qtable
    
    def setQTable(self, qtable):
        self._qtable = qtable
    
    def getStateSize(self):
        return self._env.observation_space.n

    def getActionSize(self):
        return self._env.action_space.n
    
    def reset_qtable(self):
        self._qtable = np.zeros((self._env.observation_space.n, self._env.action_space.n))

    def get_somaticMemory(self):
        return self._sm.getSomaticMemory()

    def set_somaticMemory(self, sm:dict):
        self._sm.setSomaticMemory(sm)

    def _greedy_policy(self):

        #### INTUITION
        somaticAction = self._sm.getSomaticAction(self._state) 
        
        #### SOMATIC DECISION
        if somaticAction == None:
            possibleActions = { k[0]: v for k, v in np.ndenumerate(self._qtable[self._state, :])} # get qtable state actions on dict
            action = self._sm.chooseAction(possibleActions,self._state) 
        else:
            action = somaticAction

        return action

    def _epsilon_greedy_policy(self, epsilon):
        
        random_int = random.uniform(0.0,1.0)

        # Exploitation 
        if random_int > epsilon:
            action = self._greedy_policy()
        # Exploration
        else:
            action = self._env.action_space.sample()
        return action

        
    ############## EMOTIONS ########################################################
    def _frustration(self, originalState,action,destinationState, reward):
        if originalState == destinationState:
            return -0.7

        return 0.0

    def _pain(self, originalState,action,destinationState, reward):
        if reward == -1:
            return -1.0
        
    def _euphoria(self, originalState,action,destinationState, reward):
        if reward == 1:
            return 1.0

    
