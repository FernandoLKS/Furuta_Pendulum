from envPendulum import FurutaPendulum
from simulationPendulum import Simulation
from graphSimulation import GraphSimulation
from Control import ControlSystem

import numpy as np

def main():
    ts = 0.01
    max_iterations = 1000
    initial_conditions = [np.pi, 0, 0, 0]
    
    FPendulum = FurutaPendulum()  
    #simulation = Simulation(FPendulum, ts, max_iterations, 0, initial_conditions)  
    #simulation.run_pendulum_simulation_no_control()       
    #simulationGraph = GraphSimulation()
    #simulationGraph.Plot_phase_maps()
    #simulationGraph.Plot_pendulum_simulation(simulation.get_states(), ts)
    
    control = ControlSystem(FPendulum, ts, max_iterations, initial_conditions)
    
if __name__ == '__main__':
    main()