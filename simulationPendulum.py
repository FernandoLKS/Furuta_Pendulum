from scipy.integrate import solve_ivp
import numpy as np

class Simulation():  
    '''
    Start the simulation for the Furuta pendulum.

    Inputs:
        FurutaPendulum: Object 'FurutaPendulum'
        ts: Time step for the simulation
        max_iterations: Total number of iterations for the simulation 
        torque: Initial torque applied to the system
    '''
    def __init__(self, FurutaPendulum, ts, max_iterations, torque):
        self.ts = ts
        self.max_iterations = max_iterations
        self.torque = torque       
        self.FurutaPendulum = FurutaPendulum    
        self.initialStates = self.FurutaPendulum.get_InitialConditions()
        self.states = np.zeros((self.max_iterations, 4)) 
        self.states[0, :] = self.initialStates 
        
        self.SimulatePendulumMotion() 
        
    def StateMotion(self, t, state, torque, index):
        solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, torque=0), [0, self.ts], self.states[index-1, :], method='RK45')
        self.states[index, :] = solution.y[:, -1]      
               
    def SimulatePendulumMotion(self):
        '''
        Solve ODEs at each time step.
        '''             
        for i in range(1, self.max_iterations): 
            self.StateMotion(0, self.states[i, :], 0, i)
            
    def get_FurutaPendulumInstance(self):
        '''
        Get the instance of FurutaPendulum used in the simulation.
        '''
        return self.FurutaPendulum
    
    def get_States(self):
        '''
        Get the states of the simulation.
        '''
        return self.states        