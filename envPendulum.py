import numpy as np
from scipy.integrate import solve_ivp

class FurutaPendulum():
    def __init__(self):
        self.Mb = 0.173                     # Mass of the arm (kg)
        self.Mp = 0.029                     # Mass of the pendulum (kg)
        self.r = 0.115                      # Radius of the arm (m)
        self.Lp = 0.15                      # Total length of the pendulum (m)
        self.lp = 0.075                     # Length to the pendulum's center of mass (m)
        self.Jb = 0.00094057                # Moment of inertia of the arm (kg.m^2)
        self.Jp = 0.0002175                 # Moment of inertia of the pendulum (kg.m^2)
        self.Bb = 0.0001                    # Viscous friction coefficient for the arm (N.m.s)
        self.Bp = 0.0001                    # Viscous friction coefficient for the pendulum (N.m.s)
        self.tau_m = 0                      # Applied torque (N.m)
        self.gravityForce = 9.81            # Gravitational acceleration (m/s^2) 
        
        self.theta = np.zeros(3)
        self.alpha = np.zeros(3)
    
    def Dynamic(self, t, state, u):        
        self.theta[0] = state[0]  # arm angle
        self.theta[1] = state[1]  # arm velocity angle
        self.alpha[0] = state[2]  # pendulum angle
        self.alpha[1] = state[3]  # pendulum velocity angle
        
        self.tau_m = u  
        
        # equations of motion arm and pendulum
        self.theta[2] = (self.tau_m - (self.Bb * self.theta[1]) + ((self.Mp * self.lp * self.r) * ((self.alpha[2] * np.cos(self.alpha[0])) - (self.alpha[1]**2 * np.sin(self.alpha[0]))))) / (self.Jb + (self.Mp * self.r**2))
        self.alpha[2] = ((-self.Bb * self.alpha[1]) + (self.Mp * self.lp * self.r * self.theta[2] * np.cos(self.alpha[0])) - (self.Mp * self.lp * self.gravityForce * np.sin(self.alpha[0]))) / (self.Jp + (self.Mp * self.lp**2))
                  
        derivates = np.array([self.theta[1], self.theta[2], self.alpha[1], self.alpha[2]])
        return derivates
    
    def solve_state_ODE(self, t, state, u, ts):      
        solution = solve_ivp(lambda t, state: self.Dynamic(t, state, u), [0, ts], state, method='RK45')
        return solution.y[:, -1]  