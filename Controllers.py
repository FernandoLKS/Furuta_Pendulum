import numpy as np
import pandas as pd

class Swing_up():
    '''
    Class representing the Swing-up method.
    '''
    def __init__(self, FurutaPendulum):
        self.FurutaPendulum = FurutaPendulum
    
    def EnergySystem(self):        
        Energy = (1/2 * (self.FurutaPendulum.Jb + self.FurutaPendulum.Mp + (self.FurutaPendulum.lb **2)) * self.FurutaPendulum.pendulumVelocity**2) - (self.FurutaPendulum.Mp * self.FurutaPendulum.gravityForce * self.FurutaPendulum.lb  *(np.cos(self.FurutaPendulum.pendulumAngle) + 1))
        return Energy
    
    def SignalControlTorque(self, Energy, n):
        s = (n * self.FurutaPendulum.gravityForce * np.sign(Energy) * self.FurutaPendulum.pendulumVelocity * np.cos(self.FurutaPendulum.pendulumAngle)) / self.FurutaPendulum.r

class QLearning():
    '''
    Class representing the Q-Learning algorithm.
    '''
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = None
        
    def ChooseAction(self):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            pass
        else:
            pass
        return action
    
    def Learn(self, state, action, reward, next_state):
        q_predict = self.q_table.loc[state, action]
        