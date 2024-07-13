from scipy.integrate import solve_ivp
import numpy as np

class Simulation():  

    def __init__(self, FurutaPendulum, ts, max_iterations, torque, initial_conditions):
        self.ts = ts
        self.max_iterations = max_iterations
        self.torque = torque       
        self.FurutaPendulum = FurutaPendulum   
        self.states = np.zeros((self.max_iterations, 4)) 
        
        self.set_initial_conditions(initial_conditions)  
    
    def set_initial_conditions(self, initial_conditions):
        self.FurutaPendulum.armAngle = initial_conditions[0]
        self.FurutaPendulum.armVelocity = initial_conditions[1]
        self.FurutaPendulum.pendulumAngle = initial_conditions[2]
        self.FurutaPendulum.pendulumVelocity = initial_conditions[3]
        
        self.states[0, :] = [initial_conditions[0], initial_conditions[1], initial_conditions[2], initial_conditions[3]]
    
    def get_initial_conditions(self):
        return [self.FurutaPendulum.armAngle, self.FurutaPendulum.armVelocity, self.FurutaPendulum.pendulumAngle, self.FurutaPendulum.pendulumVelocity] 
        
    def solve_state_ODE(self, t, state, u, index):
        solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, u=0), [0, self.ts], self.states[index-1, :], method='RK45')
        self.states[index, :] = solution.y[:, -1]      
               
    def run_pendulum_simulation_no_control(self):   
        for i in range(1, self.max_iterations): 
            self.solve_state_ODE(0, self.states[i, :], 0, i)
            
    def get_Furuta_pendulum_instance(self):
        return self.FurutaPendulum
    
    def get_states(self):
        return self.states        