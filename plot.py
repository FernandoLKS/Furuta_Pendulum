import numpy as np
import matplotlib.pyplot as plt
from envPendulum import FurutaPendulum

FPendulum = FurutaPendulum()

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
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.flatten()
    
    axs[0].plot(angles)
    axs[0].set_ylim(min(angles), max(angles))
    axs[0].set_title('Evolução Temporal (Ângulos do Pêndulo)')  # Temporal Evolution (Angles of Pendulum)
    axs[0].set_xlabel('Tempo (10ms)')                           # Time (10ms)
    axs[0].set_ylabel('Ângulo (rad)')                           # Angle
    
    axs[1].plot(velocities)
    axs[1].set_ylim(min(velocities), max(velocities))
    axs[1].set_title('Evolução Temporal (Velocidade Angular do Pêndulo)')   # Temporal Evolution (Velocities of Pendulum)
    axs[1].set_xlabel('Tempo (10ms)')                                       # Time (10ms)
    axs[1].set_ylabel('Velocidades angulares (rad/s)')                      # Velocity

    axs[2].plot(controls)
    axs[2].set_ylim(min(controls), max(controls))
    axs[2].set_title('Evolução Temporal (Sinais de Controle)')      # Temporal Evolution (Controls Signals)
    axs[2].set_xlabel('Tempo (10ms)')                               # Time (10ms)
    axs[2].set_ylabel('Sinais de Controle (N.m)')                   # Control Signals
    
def Plot_phase_maps(controller, id_controller): 
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))            

    # Plotando o gráfico de fase do braço
    _plot_single_phase_map(
        ax=axs[0], 
        indexX=0, 
        indexY=1, 
        labelX='Ângulo do Braço [rad]', 
        labelY='Velocidade Angular do Braço [rad/s]', 
        controller=controller, 
        id_controller=id_controller, 
        x_limits=(-np.pi, np.pi), 
        y_limits=(-10, 10)
    )

    # Plotando o gráfico de fase do pêndulo
    _plot_single_phase_map(
        ax=axs[1], 
        indexX=2, 
        indexY=3, 
        labelX='Ângulo do Pêndulo [rad]', 
        labelY='Velocidade Angular do Pêndulo [rad/s]', 
        controller=controller, 
        id_controller=id_controller, 
        x_limits=(np.pi-0.4, np.pi+0.4), 
        y_limits=(-10, 10),
        equilibrium_points=(np.array([[np.pi, 0]]), np.array([]))
    )

    plt.tight_layout()
    plt.show()

def _plot_single_phase_map(ax, indexX, indexY, labelX, labelY, controller, id_controller, x_limits, y_limits, equilibrium_points=None):
    X, Y, derivatesX, derivatesY, magnitude = _calculate_derivatives(indexX, indexY, controller, id_controller, x_limits, y_limits)

    # Plotando os pontos de equilíbrio
    if equilibrium_points:
        instables, stables = equilibrium_points
        
        if instables.size > 0:
            ax.scatter(instables[:, 0], instables[:, 1], color='r', s=10, zorder=2, label="Ponto de Equilíbrio Instável")
        if stables.size > 0:
            ax.scatter(stables[:, 0], stables[:, 1], color='g', s=10, zorder=2, label="Ponto de Equilíbrio Estável")

    # Streamplot
    ax.streamplot(X, Y, derivatesX, derivatesY, color='b', linewidth=magnitude, density=1.6)
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)
    ax.grid()

    # Adicionando legenda
    if equilibrium_points and (instables.size > 0 or stables.size > 0):
        ax.legend(loc='upper right')

def _calculate_derivatives(indexX, indexY, controller, id_controller, x_limits, y_limits):
    num_points = 100
    X, Y = np.meshgrid(np.linspace(x_limits[0], x_limits[1], num_points), 
                    np.linspace(y_limits[0], y_limits[1], num_points)) 

    derivatesX = np.zeros_like(X)
    derivatesY = np.zeros_like(Y)

    for i in range(num_points):
        for j in range(num_points):
            state = np.zeros(4)
            state[indexX] = X[i, j]     
            state[indexY] = Y[i, j]   

            # Without Control
            if id_controller == 0:  
                derivates = FPendulum.Dynamic(0, state, 0)

            # VLQR
            elif id_controller == 4:
                derivates = FPendulum.Dynamic(0, state, controller.signal_control(state, state*0))

            # PID, VLQR, SWING-UP and RL
            else:             
                derivates = FPendulum.Dynamic(0, state, controller.signal_control(state))
                    
            derivatesX[i, j] = derivates[indexX]
            derivatesY[i, j] = derivates[indexY]

    magnitude = np.sqrt(derivatesX**2 + derivatesY**2)
    magnitude /= np.max(magnitude)

    return X, Y, derivatesX, derivatesY, magnitude

def Plot_phase_map_simulation(states, control_signals):
    states = states[1:]
    derivatives = np.zeros_like(states)

    for i in range(len(states)):
        derivatives[i] = FPendulum.Dynamic(0, states[i], control_signals[i])

    magnitude_arm = np.sqrt(derivatives[:, 0]**2 + derivatives[:, 1]**2)
    magnitude_pendulum = np.sqrt(derivatives[:, 2]**2 + derivatives[:, 3]**2)
    
    scaled_derivatives = derivatives * 0.11
    scaled_derivatives[:, 0] *= magnitude_arm / np.max(magnitude_arm)  
    scaled_derivatives[:, 1] *= magnitude_arm / np.max(magnitude_arm)
    scaled_derivatives[:, 2] *= magnitude_pendulum / np.max(magnitude_pendulum)  
    scaled_derivatives[:, 3] *= magnitude_pendulum / np.max(magnitude_pendulum)

    
    fig, ax2 = plt.subplots(1, 1, figsize=(15, 8))

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
    q1 = ax1.quiver(states[:, 0], states[:, 1], scaled_derivatives[:, 0], scaled_derivatives[:, 1], magnitude_arm, angles='xy', scale_units='xy', scale=1, cmap='viridis')
    ax1.set_xlabel('Ângulo (rad)')
    ax1.set_ylabel('Velocidade Angular (rad/s)')
    #ax1.set_title('Mapa de Fase - Braço')
    ax1.set_xlim([0, 2*np.pi])
    ax1.set_ylim([-15, 15])
    #ax1.legend()
    fig.colorbar(q1, ax=ax1, label='Magnitude')"""

    
    q2 = ax2.quiver(states[:, 2], states[:, 3], scaled_derivatives[:, 2], scaled_derivatives[:, 3], magnitude_pendulum,angles='xy', scale_units='xy', scale=1, cmap='plasma')
    ax2.set_xlabel('Ângulo (rad)')
    ax2.set_ylabel('Velocidade Angular (rad/s)')
    #ax2.set_title('Mapa de Fase - Pêndulo')
    ax2.set_xlim([0, 2*np.pi])
    ax2.set_ylim([-15, 15])
    ax2.scatter(np.pi, 0, color='r', s=20, zorder=2, label="Ponto de Equilíbrio Instável")
    ax2.legend()
    fig.colorbar(q2, ax=ax2, label='Magnitude')

    plt.tight_layout()
    plt.show()