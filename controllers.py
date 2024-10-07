from fileManager import FileManipulation
from agentPendulum import discretize_state
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from envPendulum import FurutaPendulum
import numpy as np

FPendulum = FurutaPendulum()

class Swing_up():
    def __init__(self, Jp=FPendulum.Jp, Mp=FPendulum.Mp, gravityForce=FPendulum.gravityForce, lp=FPendulum.lp, k=5):
        self.Jp = Jp
        self.Mp = Mp
        self.gravityForce = gravityForce
        self.lp = lp 
        self.k = k
        
        self.Energy_equilibrium = 1/2*self.Jp*0**2 + self.Mp*self.gravityForce*self.lp*(np.cos(np.pi) - 1)  
    
    def signal_control(self, state):
        E = 1/2*self.Jp*state[3]**2 + self.Mp*self.gravityForce*self.lp*(np.cos(state[2]) - 1)     
        E0 = self.Energy_equilibrium
        
        u = (self.k*(E- E0))*np.sign(state[3]*np.cos(state[2]))
        return u

class RL_controller():
    def __init__(self):
        self.file_manager = FileManipulation()    
        self.q_table = self.file_manager.read_data_frame(file_name = 'q_table.csv')      
  
    def signal_control(self, state):
        discretized_state = discretize_state(state)
        #print(state, '-----' ,discretized_state)
        
        if np.array_str(discretized_state) in self.q_table.index:
            state_actions = self.q_table.loc[np.array_str(discretized_state)]
            u = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            u = 0
            
        return float(u)    
    
class PI_controller():
    def __init__(self, ts):
        self.Kp = 2
        self.Ki = 10 * ts
        self.previous_error = 0
        self.integral_error = 0
    
    def signal_control(self, state):
        error = np.pi - state[2]  
        
        P = self.Kp * error
        
        self.integral_error += error
        I = self.Ki * self.integral_error
        
        self.previous_error = error
        u = (P + I) 
       
        return float(u) 
    
class LQR_controller():
    def __init__(self, ts):        
        self.ts = ts
        ns = 4
        x0 = np.array([np.pi, 1e-4, np.pi, 1e-4]) # consider x0 with small dynamics (1e-4)
        u0 = np.array([0])
        self.A = Jacobian(lambda x: np.array(FPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        self.B = Jacobian(lambda u: np.array(FPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
        
        self.C = np.array([[0, 0, 1, 0]]) # output of interest is pendulum angle
        self.C = np.eye(ns)*0
        self.C[2,2]=1
        
        ny = self.C.shape[0]
        self.D = np.zeros((ny,1))
        
        self.Q = np.eye(ns)*0
        self.Q[2,2] = 1
        self.R = [[0.0001]]
        
        self.discretize()
        self.calcule_gain()
        
    def discretize(self):
        dlti = cont2discrete((self.A,self.B,self.C,self.D), self.ts)
        self.A_d = np.array(dlti[0])
        self.B_d = np.array(dlti[1])
        self.C_d = np.array(dlti[2])
        self.D_d = np.array(dlti[3])
    
    def calcule_gain(self):  
        S_d = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        self.K_lqr_d = np.linalg.inv(self.R + self.B_d.T @ S_d @ self.B_d) @ (self.B_d.T @ S_d @ self.A_d)
    
    def signal_control(self, state):
        u = -self.K_lqr_d @ state
        return float(u)
            
class VLQR_controller():
    def __init__(self, ts):      
        self.ts = ts
        
        x0 = np.array([np.pi, 1e-4, np.pi, 1e-4]) # consider x0 with small dynamics (1e-4)
        u0 = np.array([0])
        self.A = Jacobian(lambda x: np.array(FPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        self.B = Jacobian(lambda u: np.array(FPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
        self.C = np.array([[0, 0, 1, 0]]) # output of interest is pendulum angle      
               
        ns = self.A.shape[0] # number of original states     
        nu = self.B.shape[1] # number of original inputs
        ny = self.C.shape[0] # number of original outputs
        
        self.D = np.zeros((ny,1))
        self.Q = np.diag([0, 1, 0, 1, 100])
        self.R = np.array([[1]]) 
        
        self.discretize()
        
        Aaug = np.block([[self.A_d, np.zeros((ns, ny))], [self.C_d@self.A_d, np.eye(ny)]])        
        Baug = np.block([[self.B_d], [self.C_d@self.B_d]])
        
        self.A_d = Aaug
        self.B_d = Baug               
                   
        self.calcule_gain()

    def discretize(self):
        dlti = cont2discrete((self.A,self.B,self.C,self.D), self.ts)
        self.A_d = np.array(dlti[0])
        self.B_d = np.array(dlti[1])
        self.C_d = np.array(dlti[2])
        self.D_d = np.array(dlti[3])
    
    def calcule_gain(self):  
        S_d = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        self.K_lqr_d = np.linalg.inv(self.R + self.B_d.T @ S_d @ self.B_d) @ (self.B_d.T @ S_d @ self.A_d)
    
    def signal_control(self, state):
        u = -self.K_lqr_d @ state
        return float(u)

    
@staticmethod        
def Jacobian(f, x0, eps=1e-10):
    y0 = f(x0)
    m = len(y0)
    n = len(x0)

    J = np.zeros((m, n))

    for j in range(n): # for each input variable
        dx = np.zeros(n)
        dx[j] = eps
        y1 = f(x0 + dx/2)
        y0 = f(x0 - dx/2) # tustin
        J[:,j] = (y1 - y0)/eps

    return J    
    
