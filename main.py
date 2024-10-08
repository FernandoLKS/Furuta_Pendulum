from agentPendulum import Qlearning
from simulation import Simulation
from controllers import RL_controller, PI_controller, LQR_controller, VLQR_controller
import numpy as np


rl_controller = RL_controller()
pi_controller = PI_controller(ts=0.01)
lqr_controller = LQR_controller(ts=0.01)
vlqr_controller = VLQR_controller(ts=0.01)

#Qlearning(learning_rate=0.1, reward_decay=0.99, num_episodes = 100000, max_steps_per_episode = 500) # Start training agent
Simulation(rl_controller, ts=0.01, max_iterations=1500, initial_state=[0, 0, 0, 0.001]) # Simulation