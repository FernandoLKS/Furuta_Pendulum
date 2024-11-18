import numpy as np
import matplotlib.pyplot as plt

def show_rewards_and_steps(num_episodes, values_reward, values_steps):
    window_size = int(num_episodes / 20 - 1)

    moving_avg_reward = np.convolve(values_reward, np.ones(window_size) / window_size, mode='valid')
    moving_avg_steps = np.convolve(values_steps, np.ones(window_size) / window_size, mode='valid')
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    axs[0].plot(values_reward, label='Total de Recompensas')    # Total Rewards
    axs[0].set_ylim(min(values_reward), 0)
    #axs[0].set_title('Recompensas Totais por Episódio')         # Total Rewards per Episode
    axs[0].set_xlabel('Episódio')                               # Episode
    axs[0].set_ylabel('Recompensa')                             # Total Rewards
    axs[0].legend()
    
    axs[1].plot(values_steps, label='Total de Passos')          # Total Steps
    axs[1].set_ylim(0, max(values_steps))
    #axs[1].set_title('Passos Totais por Episódio')              # Total Steps per Episode
    axs[1].set_xlabel('Episódio')                               # Episode
    axs[1].set_ylabel('Passos')                                 # Total Steps
    axs[1].legend()

    axs[2].plot(range(window_size - 1, len(values_reward)), moving_avg_reward, label='Média Móvel (Recompensas)', color='red')   # Moving Average (Rewards)
    axs[2].set_ylim(min(moving_avg_reward)-1, max(moving_avg_reward)+1)
    #axs[2].set_title('Média Móvel (Recompensas)')   # Moving Average (Rewards)
    axs[2].set_xlabel('Episódio')                   # Episode
    axs[2].set_ylabel('Recompensa (Média)')         # Average Rewards
    axs[2].legend()

    axs[3].plot(range(window_size - 1, len(values_steps)), moving_avg_steps, label='Média Móvel (Passos)', color='red')     # Moving Average (Steps)
    axs[3].set_ylim(min(moving_avg_steps)-1, max(moving_avg_steps)+1)
    #axs[3].set_title('Média Móvel (Passos)')        # Moving Average (Steps)
    axs[3].set_xlabel('Episódio')                   # Episode
    axs[3].set_ylabel('Passo (Média)')              # Average Steps  
    axs[3].legend()
    
    plt.tight_layout()
    plt.close()
    return fig
    
def show_q_table(q_table, number_of_states, number_of_discrete_actions):    
    print('-------------------------------------------------------------------------------------')
    print(q_table)
    print('number of states predicted:', number_of_states)
    print('number of discrete actions:', number_of_discrete_actions)
    print('number of q-values predicted:', number_of_states * number_of_discrete_actions)
    print('number of states created in simulation', len(q_table))      
    print('-------------------------------------------------------------------------------------\n') 
    return q_table      

def show_temporal_evolution(angles, velocities, controls):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()
    
    axs[0].plot(angles)
    axs[0].set_ylim(min(angles), max(angles))
    axs[0].set_title('Evolução Temporal (Ângulos do Pêndulo)')  # Temporal Evolution (Angles of Pendulum)
    axs[0].set_xlabel('Tempo (10ms)')                           # Time (10ms)
    axs[0].set_ylabel('Ângulo')                                 # Angle
    
    axs[1].plot(velocities)
    axs[1].set_ylim(min(velocities), max(velocities))
    axs[1].set_title('Evolução Temporal (Velocidade Angular do Pêndulo)')   # Temporal Evolution (Velocities of Pendulum)
    axs[1].set_xlabel('Tempo (10ms)')                                       # Time (10ms)
    axs[1].set_ylabel('Velocidades angulares')                              # Velocity

    axs[2].plot(controls)
    axs[2].set_ylim(min(controls), max(controls))
    axs[2].set_title('Evolução Temporal (Sinais de Controle)')      # Temporal Evolution (Controls Signals)
    axs[2].set_xlabel('Tempo (10ms)')                               # Time (10ms)
    axs[2].set_ylabel('Sinais de Controle')                         # Control Signals
    