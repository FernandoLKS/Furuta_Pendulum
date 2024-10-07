import numpy as np
import pandas as pd
from envPendulum import FurutaPendulum
from graphSimulation import GraphSimulation
from fileManager import FileManipulation
import matplotlib.pyplot as plt

# Discrete states 
rangeAction = 0.6
numberActions = 7
anglePendulumVariation = 0.4

discrete_actions = np.linspace(-rangeAction, rangeAction, numberActions)

discrete_pendulum_angle_bins = [np.pi - anglePendulumVariation, np.pi - anglePendulumVariation/2, np.pi - anglePendulumVariation/4, np.pi - anglePendulumVariation/8, np.pi - anglePendulumVariation/16,  np.pi, np.pi + anglePendulumVariation/16, np.pi + anglePendulumVariation/8, np.pi + anglePendulumVariation/4, np.pi + anglePendulumVariation/2, np.pi + anglePendulumVariation]
discrete_pendulum_velocity_bins = [-np.inf, -10, -5, -2, -1,-0.5, -0.1, -0.02, 0, 0.02, 0.1, 0.5, 1, 2, 5, 10, np.inf]

number_of_states = (len(discrete_pendulum_angle_bins)-1) * (len(discrete_pendulum_velocity_bins)-1)
number_of_q_values = number_of_states * numberActions

#print(number_of_q_values)
class Qlearning:
    def __init__(self, learning_rate=0.05, reward_decay=0.95, num_episodes = 100000, max_steps_per_episode = 1000):     
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.variation_angle = 0.
        
        self.env = FurutaPendulum()
        self.q_table = pd.DataFrame(columns=discrete_actions, dtype=np.float64)
        self.file_manager = FileManipulation()  
        
        self.Training()      
    
    def ChooseAction(self, state, epsilon):                    
        discretized_state = discretize_state(state)    
        
        if np.random.uniform(0, 1) < 1 - epsilon:           
            if np.array_str(discretized_state) not in self.q_table.index:
                self.q_table.loc[np.array_str(discretized_state)] = np.zeros(len(discrete_actions))
                
            state_actions = self.q_table.loc[np.array_str(discretized_state)]
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(discrete_actions)

        return action
    
    def Learn(self, state, action, reward, next_state, done):
        if done:
            discretized_state = discretize_state(state)
            if np.array_str(discretized_state) not in self.q_table.index:
                self.q_table.loc[np.array_str(discretized_state)] = np.full(len(discrete_actions), 0, dtype=float)
                
            q_current = self.q_table.loc[np.array_str(discretized_state), action]   
                  
            q_target = reward        
        else:        
            discretized_state = discretize_state(state)
            discretized_next_state = discretize_state(next_state)
            
            if np.array_str(discretized_state) not in self.q_table.index:
                self.q_table.loc[np.array_str(discretized_state)] = np.full(len(discrete_actions), 0, dtype=float)
            
            if np.array_str(discretized_next_state) not in self.q_table.index:
                self.q_table.loc[np.array_str(discretized_next_state)] = np.full(len(discrete_actions), 0, dtype=float)
            
            q_current = self.q_table.loc[np.array_str(discretized_state), action]
            q_target = reward + self.gamma * self.q_table.loc[np.array_str(discretized_next_state), :].max()    
              
        self.q_table.loc[np.array_str(discretized_state), action] += self.learning_rate * (q_target - q_current)
    
    def reset(self):
        
        #initial_arm_angle = np.random.uniform(0, 2*np.pi)
        #initial_arm_velocity = np.random.uniform(-700, 700)
        initial_arm_angle = 0
        initial_arm_velocity = 0
        
        initial_pendulum_angle = np.random.uniform(discrete_pendulum_angle_bins[0], discrete_pendulum_angle_bins[-1])
        initial_pendulum_velocity = np.random.uniform(discrete_pendulum_velocity_bins[1]-5, discrete_pendulum_velocity_bins[-2]+5)
        
        reset_state = np.array([[initial_arm_angle, initial_arm_velocity, initial_pendulum_angle, initial_pendulum_velocity]])
        
        return reset_state
    
    def step(self, action, states):
        
        ref_state = np.array([0, 0, np.pi, 0])
        current_state = np.array(states[-1])          
        next_state = self.env.solve_state_ODE(0, current_state, action, ts=0.01)                                    
                       
        state_variance = current_state - ref_state
            
        Q = [0, 0, 10, 1]
        state_variance = (state_variance * Q)**2
            
        reward = -(np.sum(state_variance))  
                        
        Condition1 = np.abs(next_state[2]) > np.pi+anglePendulumVariation or np.abs(next_state[2]) < np.pi-anglePendulumVariation
        #Condition2 = np.abs(current_state[1]) > np.pi+1 or np.abs(current_state[1]) < np.pi-1
        
        done = Condition1        
        
        return next_state, reward, done
        
    def functionE_greedy(self, current_episode):
        # e_greddy linear
        epsilon = 1 - current_episode / self.num_episodes
        
        # e_greddy exponencial
        #epsilon = 1 - np.exp(-3*(self.num_episodes - current_episode - 1)/self.num_episodes)            
        
        # e_greddy constant
        #epsilon = 0.8
            
        #if current_episode/self.num_episodes < 0.5: epsilon = 0.9
        #elif current_episode/self.num_episodes < 0.6: epsilon = 0.5
        #elif current_episode/self.num_episodes < 0.7: epsilon = 0.2
        #else: epsilon = 0.0
        
        return epsilon    
    
    def Training(self):        
        simulation = GraphSimulation()
        self.list_total_rewards = []
        self.list_steps_by_episode = []

        for episode in range(self.num_episodes):
            if (episode+1) % (self.num_episodes/20) == 0:
                self.file_manager.save_graphic(show_rewards(self.num_episodes, self.list_total_rewards, self.list_steps_by_episode), 'graphic.png')  
                self.file_manager.save_data_frame(self.q_table, 'q_table.csv')
            
            states = self.reset()
        
            total_reward = 0
            done = False
            self.epsilon = self.functionE_greedy(episode)
  
            for step in range(self.max_steps_per_episode):  
                current_state = states[-1]               
                action = self.ChooseAction(current_state, self.epsilon)      
                               
                next_state, reward, done = self.step(action, states)                   
                                                        
                self.Learn(current_state, action, reward, next_state, done)                
                states = np.vstack((states, next_state))
                
                total_reward += reward
                
                if done:
                    break
            
            self.list_total_rewards.append(total_reward)
            self.list_steps_by_episode.append(step)
            
            print(f"Episode {episode+1}/{self.num_episodes} - Total Reward: {total_reward:.4f} - Total steps: {step}")   
        print("\nTraining complete.")
        
        fig = show_rewards(self.num_episodes, self.list_total_rewards, self.list_steps_by_episode)  
        show_q_table(self.q_table)
        self.file_manager.save_data_frame(self.q_table, 'q_table.csv')
        self.file_manager.save_graphic(fig, 'graphic.png')
        
        simulation.Plot_pendulum_simulation(states, 0.01)
        

    
def discretize_state(state):
    state[0] = state[0] % (2 * np.pi)
    state[2] = state[2] % (2 * np.pi)

    bins = [discrete_pendulum_angle_bins, discrete_pendulum_velocity_bins]
    discretized_state = np.empty(len(bins), dtype=int)

    #intervals = []

    for i in range(len(bins)):
        discretized_state[i] = np.digitize(state[i+2], bins[i])
        
        #lower_bound = bins[i][discretized_state[i] - 1] if discretized_state[i] > 0 else None
        #upper_bound = bins[i][discretized_state[i]] if discretized_state[i] < len(bins[i]) else None
        
        #intervals.append([lower_bound, upper_bound])

    #print(f"State: {state} = {intervals}")

    return discretized_state 

def show_rewards(num_episodes, values_reward, values_steps):
    window_size = int(num_episodes / 20 - 1)

    moving_avg_reward = np.convolve(values_reward, np.ones(window_size) / window_size, mode='valid')
    moving_avg_steps = np.convolve(values_steps, np.ones(window_size) / window_size, mode='valid')
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    axs[0].plot(values_reward, label='Total Rewards')
    axs[0].set_ylim(min(values_reward), 0)
    axs[0].set_title('Total Rewards per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Rewards')
    axs[0].legend()
    
    axs[1].plot(values_steps, label='Total Steps')
    axs[1].set_ylim(0, max(values_steps))
    axs[1].set_title('Total Steps per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Total Steps')
    axs[1].legend()

    axs[2].plot(range(window_size - 1, len(values_reward)), moving_avg_reward, label='Moving Average (Rewards)', color='red')
    axs[2].set_ylim(min(moving_avg_reward)-1, max(moving_avg_reward)+1)
    axs[2].set_title('Moving Average (Rewards)')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Rewards')
    axs[2].legend()

    axs[3].plot(range(window_size - 1, len(values_steps)), moving_avg_steps, label='Moving Average (Steps)', color='red')
    axs[3].set_ylim(min(moving_avg_steps)-1, max(moving_avg_steps)+1)
    axs[3].set_title('Moving Average (Steps)')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Average Steps')
    axs[3].legend()
    
    plt.tight_layout()
    plt.close()
    return fig
    
def show_q_table(q_table):    
    #max_value = self.q_table.max().max()
    print('-------------------------------------------------------------------------------------')
    print(q_table)
    print('number of states predicted:', number_of_states)
    print('number of q-values predicted:', number_of_states * len(discrete_actions))
    print('number of states created in simulation', len(q_table))      
    print('-------------------------------------------------------------------------------------\n') 
    return q_table      