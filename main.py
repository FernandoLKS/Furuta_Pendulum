from envPendulum import FurutaPendulum
from simulationPendulum import Simulation, GraphSimulation
from Controllers import QLearning

def main():
    FurutaPendulum = FurutaPendulum()
    simulation = Simulation(FurutaPendulum, 0.01, 1000, 0)
    simulationGraph = GraphSimulation(simulation)  
    
    #RLAlgorithm = QLearning()
    
if __name__ == '__main__':
    main()