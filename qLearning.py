import numpy as np
import pandas as pd

class QLearning():
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(colums=self.actions, dtype=np.float64)
        
    def choose_action(self):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            pass
        else:
            pass
        return action
    
    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table.loc[state, action]
        