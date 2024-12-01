import numpy as np
from scipy.integrate import solve_ivp    
from fileManager import FileManipulation
from envPendulum import FurutaPendulum
from graphSimulation import GraphDynamicSimulation
from controllers import Swing_up
from plot import show_temporal_evolution, Plot_phase_maps, Plot_phase_map_simulation
from controllers import VLQR_controller
from scipy.optimize import minimize

NO_CONTROLLER = 0
RL_CONTROLLER = 1
PID_CONTROLLER = 2
LQR_CONTROLLER = 3
VLQR_CONTROLLER = 4

class Simulation():
    def __init__(self, Controller, id_controller, ts=0.01, max_iterations=2000, initial_state=[0, 0, 0, 0.001]):
        self.file_manager = FileManipulation()    
        self.FPendulum = FurutaPendulum()
        self.graphDynamicSimulation = GraphDynamicSimulation()               
        
        self.controller = Controller
        self.id_controller = id_controller
        self.ts = ts
        self.max_iterations = max_iterations
        self.swing_up = Swing_up(k=0.5)
        
        self.state = np.zeros((max_iterations, 4)) 
        self.state[0, :] = initial_state
        
        self.signal_control = []
        self.erro_simulation = []
        self.signal_control_estabilization = []
        self.erro_estabilization = []

        self.u = 0 
        self.variation_angle = 0.4    # threshold between swing up and stabilizing controller     
        self.control()            
        
    def control(self):
        # noise adition [0.99 - 1.01  mean = 1]   
        min_noise = 0.99
        max_noise = 1.01

        noise_list = np.random.normal(loc=1, scale=0.1, size=3000)
        normalized_noise = (noise_list - noise_list.min()) / (noise_list.max() - noise_list.min())
        noise_list = normalized_noise * (max_noise - min_noise) + min_noise

        for i in range(1, self.max_iterations): 
            # noise
            #self.state[i-1, :] *= noise_list[i]          
    
            solution = solve_ivp(lambda t, state: self.FPendulum.Dynamic(t, state, self.u), [0, self.ts], self.state[i-1, :], method='RK45')
            self.state[i, :] = solution.y[:, -1]

            # Angle normalize (0 - 2*pi)
            self.state[i, 0] = self.state[i, 0] % (2 * np.pi)  
            self.state[i, 2] = self.state[i, 2] % (2 * np.pi)   
            
            self.error = [np.sin(np.pi) - np.sin(self.state[i, 2]), np.cos(np.pi) - np.cos(self.state[i, 2])] # pendulum angle decomposed in sin/cos since 0 rad = 2*pi rad
            self.error = self.error[0] + self.error[1]    
                        
            Condition = np.abs(self.state[i, 2]) > np.pi+self.variation_angle or np.abs(self.state[i, 2]) < np.pi-self.variation_angle  

            """if(i == 1000):
                print("AAAAAAAAAAAAAAAAAAAAAAAa")
                self.state[i, 3] = 4  

            if(i == 2000):
                print("AAAAAAAAAAAAAAAAAAAAAAAa")
                self.state[i, 3] = -4 """
               
            # SWING-UP controller              
            if (Condition):                
                self.u = self.swing_up.signal_control(self.state[i, :])   
                print(f'*** SWING UP *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')           
                   
            # STABILIZING controller
            else:      
                if(self.id_controller == RL_CONTROLLER):      
                    self.u = self.controller.signal_control(self.state[i, :])       
                    print(f'*** STABILIZING CONTROL (RL) *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')   
                    
                elif(self.id_controller == PID_CONTROLLER ):
                    #self.u = self.controller.signal_control(self.state[i, :])  
                    #print(f'*** STABILIZING CONTROL (LQR) *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')  

                    self.u = self.controller.signal_control(self.state[i, :])       
                    print(f'*** STABILIZING CONTROL (PID) *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')                     
                    
                elif(self.id_controller == LQR_CONTROLLER):                                      
                    self.u = self.controller.signal_control(self.state[i, :])  
                    print(f'*** STABILIZING CONTROL (LQR) *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')  
                     
                elif(self.id_controller == VLQR_CONTROLLER):                       
                    self.u += self.controller.signal_control(self.state[i, :], self.state[i-1, :])
                    self.u = saturation_signal(self.u, 0.6)                      
                    print(f'*** STABILIZING CONTROL (VLQR) *** Control Signal: {self.u:.8f} -- Error: {self.error:.6f}')   
                
                self.erro_estabilization.append(self.error)
                self.signal_control_estabilization.append(self.u)               

            self.erro_simulation.append(self.error)
            self.signal_control.append(self.u)

        msa_e = np.mean((np.array(self.erro_estabilization)))
        mse_e = np.mean((np.array(self.erro_estabilization)) ** 2)
        c = np.sum((np.array(self.signal_control)) ** 2)

        print(f'*** MSA (PENDULUM ANGLE) {msa_e:.8f} *** MSE (PENDULUM ANGLE) {mse_e:.8f} *** ERRO ACUMULATE (CONTROL SIGNAL) {c:.8f}')        
            
        show_temporal_evolution(self.state[:, 2], self.state[:, 3], self.signal_control)   

        self.graphDynamicSimulation.Plot_pendulum_simulation(self.state, self.ts)    

        Plot_phase_maps(self.controller, self.id_controller) 
        Plot_phase_map_simulation(self.state, self.signal_control)
        
@staticmethod  
def saturation_signal(x, sat):
    if x < sat and x > -sat:
        return x
    elif x <= -sat:
        return -sat
    elif x >= sat:
        return sat     
    