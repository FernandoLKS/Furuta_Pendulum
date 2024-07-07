from simulationPendulum import Simulation
from qLearning import QLearning

def main():
    simulation = Simulation(0.01, 1000, 0)
    simulation.Execute()    
    
    #RLAlgorithm = QLearning()
    
if __name__ == '__main__':
    main()