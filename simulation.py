import numpy as np
from scipy.integrate import solve_ivp    
from fileManager import FileManipulation
from envPendulum import FurutaPendulum
from graphSimulation import GraphSimulation
from controllers import Swing_up
from plot import show_temporal_evolution

class Simulation():
    def __init__(self, Controller, ts=0.01, max_iterations=2000, initial_state=[0, 0, 0, 0.001]):
        self.file_manager = FileManipulation()    
        self.FPendulum = FurutaPendulum()
        self.graphSimulation = GraphSimulation()    
        self.controller = Controller    
        
        self.ts = ts
        self.max_iterations = max_iterations
        self.swing_up = Swing_up(k=5)
        
        self.state = np.zeros((max_iterations, 4)) 
        self.state[0, :] = initial_state
        self.signal_control = []
        
        self.u_saturation = 0.6
        self.u = 0 
        self.variation_angle = 0.4    
        
        self.control()
    
    def control(self):
        for i in range(1, self.max_iterations):           
    
            solution = solve_ivp(lambda t, state: self.FPendulum.Dynamic(t, state, self.u), [0, self.ts], self.state[i-1, :], method='RK45')
            self.state[i, :] = solution.y[:, -1]
            self.state[i, 0] = self.state[i, 0] % (2 * np.pi)  
            self.state[i, 2] = self.state[i, 2] % (2 * np.pi)   
            
            self.error = [np.sin(np.pi) - np.sin(self.state[i, 2]), np.cos(np.pi) - np.cos(self.state[i, 2])] # pendulum angle decomposed in sin/cos since 0 rad = 2*pi rad
            self.error = self.error[0] + self.error[1]     
                        
            Condition = np.abs(self.state[i, 2]) > np.pi+self.variation_angle or np.abs(self.state[i, 2]) < np.pi-self.variation_angle                             
            
            # SWING-UP controller              
            if (Condition):                
                self.u = self.swing_up.signal_control(self.state[i, :])        
                # saturation of control signal    
                self.u = self.saturation_signal(self.u, self.u_saturation)    
                print(f'*** SWING UP *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')           
                   
            # STABILIZING controller
            else:            
                self.u = self.controller.signal_control(self.state[i, :])
                # saturation of control signal    
                self.u = self.saturation_signal(self.u, self.u_saturation)        
                print(f'*** STABILIZING CONTROL *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')    
                
            self.signal_control.append(self.u)      
            
        show_temporal_evolution(self.state[:, 2], self.state[:, 3], self.signal_control)          
        self.graphSimulation.Plot_pendulum_simulation(self.state, self.ts) 
        
        
    @staticmethod  
    def saturation_signal(x, sat):
        if x <= sat and x >= -sat:
            return x
        elif x < -sat:
            return -sat
        else:
            return sat     