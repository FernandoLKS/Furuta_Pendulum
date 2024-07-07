from envPendulum import FurutaPendulum
from simulationPendulum import Simulation
from qLearning import QLearning

def main():
    FPendulum = FurutaPendulum()
    Simulation(FPendulum, 0.01, 1000, 0)  
    
    #RLAlgorithm = QLearning()
    
if __name__ == '__main__':
    main()