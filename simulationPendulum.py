from scipy.integrate import solve_ivp
import numpy as np

class Simulation():  

    def __init__(self, FurutaPendulum, ts, max_iterations, torque, armAngleInit=0, armVelocityInit=0, pendulumAngleInit=0, pendulumVelocityInit=0):
        self.ts = ts
        self.max_iterations = max_iterations
        self.torque = torque       
        self.FurutaPendulum = FurutaPendulum   
        self.states = np.zeros((self.max_iterations, 4)) 
        
        self.Set_InitialConditions(armAngleInit, armVelocityInit, pendulumAngleInit, pendulumVelocityInit)  
    
    def Set_InitialConditions(self, armAngle, armVelocity, pendulumAngle, pendulumVelocity):
        self.FurutaPendulum.armAngle = armAngle
        self.FurutaPendulum.armVelocity = armVelocity
        self.FurutaPendulum.pendulumAngle = pendulumAngle
        self.FurutaPendulum.pendulumVelocity = pendulumVelocity
        
        self.states[0, :] = [armAngle, armVelocity, pendulumAngle, pendulumVelocity]
    
    def get_InitialConditions(self):
        return [self.FurutaPendulum.armAngle, self.FurutaPendulum.armVelocity, self.FurutaPendulum.pendulumAngle, self.FurutaPendulum.pendulumVelocity] 
        
    def Solve_state_ODE(self, t, state, u, index):
        solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, u=0), [0, self.ts], self.states[index-1, :], method='RK45')
        self.states[index, :] = solution.y[:, -1]      
               
    def Run_simulation_pendulum(self):   
        for i in range(1, self.max_iterations): 
            self.Solve_state_ODE(0, self.states[i, :], 0, i)
            
    def get_FurutaPendulumInstance(self):
        return self.FurutaPendulum
    
    def get_States(self):
        return self.states        