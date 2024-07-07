from envPendulum import FurutaPendulum
from simulationPendulum import Simulation
from graphSimulation import GraphSimulation
from Controllers import Swing_up, QLearning

import numpy as np

def main():
    FPendulum = FurutaPendulum()
    FPendulum.set_InitialConditions(np.pi, 0, 4/3 * np.pi, 0)
    
    simulation = Simulation(FPendulum, 0.01, 1000, 0)
    
    simulationGraph = GraphSimulation(simulation)
    simulationGraph.PlotPhaseMaps()
    simulationGraph.PlotSimulation()
    
    swingUp = Swing_up(FPendulum)

    
if __name__ == '__main__':
    main()