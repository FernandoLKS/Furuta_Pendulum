import numpy as np
from scipy.integrate import solve_ivp
import scipy
from graphSimulation import GraphSimulation
from Controllers import Swing_up, VLQR, LQR, PI

class ControlSystem():
    def __init__(self, FurutaPendulum, ts, max_iterations, initial_conditions):
        self.FurutaPendulum = FurutaPendulum  
        self.graphSimulation = GraphSimulation() 
           
        self.state = np.zeros((max_iterations, 4)) 
        self.state[0, :] = initial_conditions
        
        self.u=0            
        self.error = 0    # current error
        self.errorp = 0   # past error
        
        self.statei = np.zeros((max_iterations, 1)) 
        
        self.Control_Estrategy(ts, max_iterations)
        
    def Control_Estrategy(self, ts, max_iterations): 
        swing_up = self.create_swing_up(k=6)
        vlqr = self.create_vlqr(ts)
        lqr = self.create_lqr(ts)
        pi = self.create_pi(ts)
        
        for i in range(1, max_iterations):            
            # System
            solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, self.u), [0, ts], self.state[i-1, :], method='RK45')
            self.state[i, :] = solution.y[:, -1]
            
            # Observation
            self.error = [np.sin(np.pi) - np.sin(self.state[i, 2]), np.cos(np.pi) - np.cos(self.state[i, 2])] # pendulum angle decomposed in sin/cos since 0 rad = 2*pi rad
            self.error = self.error[0] + self.error[1]
            
            # Control            
            self.Switch_controllers(0.5, swing_up, pi, i, self.u)
                
            # control saturation           
            self.u = self.satng(self.u, 0.6)            
            print(self.error, self.u)
            
            self.errorp = self.error
        
        self.graphSimulation.Plot_pendulum_simulation(self.state, ts)    
        
    def Switch_controllers(self, error_limit, swing_up, controller, i, u):
        if np.abs(self.error) <= error_limit:
            print('*** STABILIZING CONTROL ***')
            #vstates = np.hstack([self.state[i,:] - self.state[i-1,:], self.statei[i,:] - self.statei[i-1,:]])
            self.u = controller.get_signal_control(self.error, self.errorp, u)
        
        # swingup controller
        if np.abs(self.error) > error_limit:
            print('*** SWING UP ***')
            self.u = swing_up.get_signal_control(i, self.state)
        
    def create_swing_up(self, k):
        Jp = self.FurutaPendulum.Jp
        Mp = self.FurutaPendulum.Mp
        gravityForce = self.FurutaPendulum.gravityForce
        lp = self.FurutaPendulum.lp
        
        swing_up = Swing_up(Jp, Mp, gravityForce, lp, k)
        return swing_up
        
    def create_vlqr(self, ts):
        x0 = np.array([np.pi, 1e-4, np.pi, 1e-4]) # consider x0 with small dynamics (1e-4)
        u0 = np.array([0])
        A = self.Jacobian(lambda x: np.array(self.FurutaPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        B = self.Jacobian(lambda u: np.array(self.FurutaPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
        C = np.array([[0, 0, 1, 0]]) # output of interest is pendulum angle
        #C = np.eye(ns)*0
        #C[2,2]=1
        ny = C.shape[0]
        D = np.zeros((ny,1))
        Q = np.diag([0, 1, 0, 1, 100]) # considering velocities and pendulum angle error
        R = np.array([[0.1]])
        
        vlqr = VLQR(A, B, C, D, ts, Q, R) 
        return vlqr
        
    def create_lqr(self, ts):
        ns = 4
        x0 = np.array([np.pi, 1e-4, np.pi, 1e-4]) # consider x0 with small dynamics (1e-4)
        u0 = np.array([0])
        A = self.Jacobian(lambda x: np.array(self.FurutaPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        B = self.Jacobian(lambda u: np.array(self.FurutaPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
        C = np.array([[0, 0, 1, 0]]) # output of interest is pendulum angle
        C = np.eye(ns)*0
        C[2,2]=1
        ny = C.shape[0]
        D = np.zeros((ny,1))
        
        Q = np.eye(ns)*0
        Q[2,2] = 1
        R = [[0.0001]]
        
        lqr = LQR(A, B, C, D, ts, Q, R)      
        return lqr  
    
    def create_pi(self, ts):
        pi = PI(ts)
        
        return pi
               
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
    
    @staticmethod   
    def Controlabillity(A, B, ns, nu):
        rankA = np.linalg.matrix_rank(A)
        rankB = np.linalg.matrix_rank(B)
        
        if rankB < rankA:
            print(f'System is underactuated {rankB} < {rankA}')
        
        controlabillity = np.reshape([A**i@B for i in range(ns)], (ns, ns*nu))
        rankCTRB = np.linalg.matrix_rank(controlabillity)

        if rankCTRB == ns:
            print('System is controllable')
        else:
            print('System is non controllable')      
            
    @staticmethod  
    def lqr(A,B,Q,R):
        # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B) 
        # and cost function parameterized by (Q,R)
        
        # solve DARE:
        M=scipy.linalg.solve_discrete_are(A,B,Q,R)

        # K=(B'MB + R)^(-1)*(B'MA)
        #return np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))
        return np.linalg.inv(R + B.T @ M @ B) @ (B.T @ M @ A)     
    
    @staticmethod  
    def satng(x, sat):
        if x < sat and x > -sat:
            return x
        elif x < -sat:
            return -sat
        else:
            return sat  
