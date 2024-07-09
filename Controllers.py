import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from scipy.linalg import solve_continuous_are, solve_discrete_are

class ControlSystem():
    def __init__(self, FurutaPendulum):
        self.FurutaPendulum = FurutaPendulum
        
    def Control_Estrategy():
        pass
        
    def SwitchControl():
        pass
        
# Full State Feedback
class FSFB:
    def __init__(self, A, B, C, D, dt: float):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        
        # number of states
        self.ns = A.shape[0]
        # number of inputs
        self.nu = B.shape[1]
        # number of outputs
        self.ny = C.shape[0]

        self.discretize()

    def discretize(self):
        dlti = scipy.signal.cont2discrete((self.A,self.B,self.C,self.D), self.dt)
        self.A_d = np.array(dlti[0])
        self.B_d = np.array(dlti[1])
        self.C_d = np.array(dlti[2])
        self.D_d = np.array(dlti[3])
                
    # TODO:
    def pole_placement(self, poles):
        pass
    
    # TODO:
    def feedback(self, state):
        pass

class LQR(FSFB):
    def __init__(self, A, B, C, D, dt: float, Q, R):
        super().__init__(A, B, C, D, dt)
        self.Q = Q
        self.R = R
        
        self.calculate_K()
        
    def calculate_K(self):
        S = solve_continuous_are(self.A, self.B, self.Q, self.R)
        S_d = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)

        self.K_lqr = np.linalg.inv(self.R) @ self.B.T @ S
        self.K_lqr_d = np.linalg.inv(self.R + self.B_d.T @ S_d @ self.B_d) @ (self.B_d.T @ S_d @ self.A_d)
        
    def closed_loop_poles(self):
        return np.linalg.eig(self.A - self.B*self.K_lqr).eigenvalues
    
    def discrete_closed_loop_poles(self):
        return np.linalg.eig(self.A_d - self.B_d*self.K_lqr_d).eigenvalues

    def feedback(self, state):
        u = -self.K_lqr @ state
        return u

    def discrete_feedback(self, state):
        u = -self.K_lqr_d @ state
        return u
    
class VLQR(LQR):
    def __init__(self, A, B, C, D, dt: float, Q, R):
        # number of original states
        ns = A.shape[0]
        # number of original inputs
        nu = B.shape[1]
        # number of original outputs
        ny = C.shape[0]
        
        # Augment matrices of Velocity LQR
        # https://sites.engineering.ucsb.edu/~jbraw/jbrweb-archives/tech-reports/twmcc-2001-01.pdf
        # TODO: numerically unstable?
        '''Aaug = np.block([[A, np.zeros((ns, ny))], [C@A, np.eye(ny)]])
        
        Baug = np.block([[B], [C@B]])
        
        # initialize LQR with augmented matrices
        super().__init__(Aaug, Baug, C, D, dt, Q, R)'''
        
        # for reasons of numerical stability, initialize FSFB and discretize the regular system
        super().__init__(A, B, C, D, dt, Q[:ns,:ns], R[:nu,:nu])
        
        # continuous augmented matrices
        Aaug = np.block([[self.A, np.zeros((ns, ny))], [self.C@self.A, np.eye(ny)]])        
        Baug = np.block([[self.B], [self.C@self.B]])
        
        self.A = Aaug
        self.B = Baug
        
        # discrete augmented matrices
        Aaug = np.block([[self.A_d, np.zeros((ns, ny))], [self.C_d@self.A_d, np.eye(ny)]])        
        Baug = np.block([[self.B_d], [self.C_d@self.B_d]])
        
        self.A_d = Aaug
        self.B_d = Baug
        
        # store true Q and R matrices
        self.Q = Q
        self.R = R
        
        # recalculate gains
        self.calculate_K()
        
class PI():
    pass

class Swing_up():
    def __init__(self, FurutaPendulum):
        self.FurutaPendulum = FurutaPendulum
    
    def EnergySystem(self):        
        Energy = (1/2 * (self.FurutaPendulum.Jb + self.FurutaPendulum.Mp + (self.FurutaPendulum.lb **2)) * self.FurutaPendulum.pendulumVelocity**2) - (self.FurutaPendulum.Mp * self.FurutaPendulum.gravityForce * self.FurutaPendulum.lb  *(np.cos(self.FurutaPendulum.pendulumAngle) + 1))
        return Energy
    
    def SignalControlTorque(self, Energy, numberBalances):
        s = (numberBalances * self.FurutaPendulum.gravityForce * np.sign(Energy) * self.FurutaPendulum.pendulumVelocity * np.cos(self.FurutaPendulum.pendulumAngle)) / self.FurutaPendulum.r

    def satng(x, sat):
        if x < sat and x > -sat:
            return x
        elif x < -sat:
            return -sat
        else:
            return sat
    
class QLearning():
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = None
        
    def ChooseAction(self):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            pass
        else:
            pass
        return action
    
    def Learn(self, state, action, reward, next_state):
        q_predict = self.q_table.loc[state, action]
        
'''       
def lqr(A,B,Q,R):
    # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B) 
    # and cost function parameterized by (Q,R)
    
    # solve DARE:
    M=scipy.linalg.solve_discrete_are(A,B,Q,R)

    # K=(B'MB + R)^(-1)*(B'MA)
    #return np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))
    return np.linalg.inv(R + B.T @ M @ B) @ (B.T @ M @ A)

# F : Rn -> Rm
#J = m x n
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
      
def satng(x, sat):
    if x < sat and x > -sat:
        return x
    elif x < -sat:
        return -sat
    else:
        return sat
'''         