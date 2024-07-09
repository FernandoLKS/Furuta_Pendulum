from envPendulum import FurutaPendulum
from simulationPendulum import Simulation
from graphSimulation import GraphSimulation
from Controllers import Swing_up, QLearning

import numpy as np

def main():
    FPendulum = FurutaPendulum()    
    
    simulation = Simulation(FPendulum, 0.01, 1000, 0, np.pi, 0, 4/3 * np.pi, 0)  
    simulation.Run_simulation_pendulum()   
    
    simulationGraph = GraphSimulation(simulation)
    #simulationGraph.Plot_phase_maps()
    simulationGraph.Plot_pendulum_simulation()
    
    swingUp = Swing_up(FPendulum)
    
if __name__ == '__main__':
    main()