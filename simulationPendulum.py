from scipy.integrate import solve_ivp
import numpy as np

class Simulation():  

    def __init__(self, FurutaPendulum, ts, max_iterations, torque):
        self.ts = ts
        self.max_iterations = max_iterations
        self.torque = torque       
        self.FurutaPendulum = FurutaPendulum   
        self.states = np.zeros((self.max_iterations, 4)) 
        self.states[0, :] = self.FurutaPendulum.get_InitialConditions() 
        
        self.Run_simulation(self.states) 
        
    def Solve_state_ODE(self, t, state, torque, index):
        solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, torque=0), [0, self.ts], self.states[index-1, :], method='RK45')
        self.states[index, :] = solution.y[:, -1]      
               
    def Run_simulation(self, states):   
        for i in range(1, self.max_iterations): 
            self.Solve_state_ODE(0, states[i, :], 0, i)
            
    def get_FurutaPendulumInstance(self):
        return self.FurutaPendulum
    
    def get_States(self):
        return self.states        