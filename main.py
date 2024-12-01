from agentPendulum import Qlearning
from simulation import Simulation
from controllers import Swing_up, RL_controller, PID_controller, LQR_controller, VLQR_controller

NO_CONTROL = 0
RL_CONTROLLER = 1
PID_CONTROLLER = 2
LQR_CONTROLLER = 3
VLQR_CONTROLLER = 4
SWINGUP_CONTROLLER = 5

swingup_controller = Swing_up(k=0.5)
rl_controller = RL_controller()
pid_controller = PID_controller(ts=0.01)
lqr_controller = LQR_controller(ts=0.01)
vlqr_controller = VLQR_controller(ts=0.01)

#Qlearning(learning_rate=0.1, reward_decay=0.99, num_episodes = 40000, max_steps_per_episode = 500) # Start training agent

#Simulation(swingup_controller, SWINGUP_CONTROLLER, ts=0.01, max_iterations=3000, initial_state=[0, 0, 0, 0.001]) # Simulation
#Simulation(rl_controller, RL_CONTROLLER, ts=0.01, max_iterations=3000, initial_state=[0, 0, 0, 0.001]) # Simulation
Simulation(pid_controller, PID_CONTROLLER, ts=0.01, max_iterations=3000, initial_state=[0, 0, 0, 0.001]) # Simulation
#Simulation(lqr_controller, LQR_CONTROLLER, ts=0.01, max_iterations=3000, initial_state=[0, 0, 0, 0.001]) # Simulation
#Simulation(vlqr_controller, VLQR_CONTROLLER, ts=0.01, max_iterations=3000, initial_state=[0, 0, 0, 0.001]) # Simulation

