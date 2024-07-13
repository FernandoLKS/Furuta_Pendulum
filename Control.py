import numpy as np
from scipy.integrate import solve_ivp
import scipy
from graphSimulation import GraphSimulation
from Controllers import Swing_up, VLQR, LQR

class ControlSystem():
    def __init__(self, FurutaPendulum, ts, max_iterations, initial_conditions):
        self.FurutaPendulum = FurutaPendulum  
        self.graphSimulation = GraphSimulation()
        self.functions = AuxiliarFunctions    
        
        self.Control_Estrategy(ts, max_iterations, initial_conditions)
        
    def create_swing_up(self):
        swing_up = Swing_up(self.FurutaPendulum)
        return swing_up
        
    def create_vlqr(self, ts):
        x0 = np.array([np.pi, 1e-4, np.pi, 1e-4]) # consider x0 with small dynamics (1e-4)
        u0 = np.array([0])
        A = self.functions.Jacobian(lambda x: np.array(self.FurutaPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        B = self.functions.Jacobian(lambda u: np.array(self.FurutaPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
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
        A = self.functions.Jacobian(lambda x: np.array(self.FurutaPendulum.Dynamic(0, x, u0)), x0)  # linearization of states
        B = self.functions.Jacobian(lambda u: np.array(self.FurutaPendulum.Dynamic(0, x0, u)), u0)  # linearization of inputs
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
        
    def Control_Estrategy(self, ts, max_iterations, initial_conditions): 
        swing_up = self.create_swing_up()
        vlqr = self.create_vlqr(ts)
        lqr = self.create_lqr(ts)
               
        state = np.zeros((max_iterations, 4)) 
        state[0, :] = initial_conditions
        u=0    
        
        error = 0    # current error
        errorp = 0   # past error
        
        statei = np.zeros((max_iterations, 1)) 
        
        for i in range(1, max_iterations):            
            ### System
            solution = solve_ivp(lambda t, state: self.FurutaPendulum.Dynamic(t, state, u), [0, ts], state[i-1, :], method='RK45')
            state[i, :] = solution.y[:, -1]
            
            ## Observation
            error = [np.sin(np.pi) - np.sin(state[i, 2]), np.cos(np.pi) - np.cos(state[i, 2])] # pendulum angle decomposed in sin/cos since 0 rad = 2*pi rad
            
            ### Control
            
            # error integral for velocity LQR
            solution = solve_ivp(lambda t, statei: np.array(np.pi - state[i,2]), [0, ts], statei[i-1, :], method='RK45') # set-point tracking
            #solution = solve_ivp(lambda t, statei: np.array(error[0]+error[1]), [0, ts], statei[i-1, :], method='RK45')  # pendulum angle decomposition
            statei[i, :] = solution.y[:, -1]
            
            # PI
            error = error[0] + error[1]
            #error = np.pi - state[i,2]
            '''if np.abs(error) <= 0.01:
                print('*** STABILIZING CONTROL ***')
                kp, ki = 4, 0.1
                du = kp*(error - errorp) + ki*error*ts
                u += du
                ##u = kp*error + ki*statei[i,:] # positional control
                
                errorp = error'''
            
            # velocity LQR
            if np.abs(error) <= 0.1:
                print('*** STABILIZING CONTROL ***')
                vstates = np.hstack([state[i,:] - state[i-1,:], statei[i,:] - statei[i-1,:]])
                u += vlqr.discrete_feedback(vstates)
            
            # LQR variable change approach
            # offset states by the desired references
            '''rstates = (state[i,:] - np.array([np.pi, 0, np.pi, 0]))
            u = lqr.discrete_feedback(rstates)'''
            
            # swingup controller
            if np.abs(error) > 0.1:
                k = 6
                E0 = swing_up.get_energy_equilibrium()
                E = swing_up.get_energy_current(i, state)
                #u = k*(E - E0)*state[i,3]*np.cos(state[i,2])
                u = self.functions.satng(swing_up.get_signal_control(k, i, state, E, E0), 0.6)
                print('*** SWING UP ***')
                
            # control saturation           
            u = self.functions.satng(u, 0.6)
            
            print(error, u)
        
        self.graphSimulation.Plot_pendulum_simulation(state, ts) 
        
class AuxiliarFunctions():    
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
    
    def lqr(A,B,Q,R):
        # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B) 
        # and cost function parameterized by (Q,R)
        
        # solve DARE:
        M=scipy.linalg.solve_discrete_are(A,B,Q,R)

        # K=(B'MB + R)^(-1)*(B'MA)
        #return np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))
        return np.linalg.inv(R + B.T @ M @ B) @ (B.T @ M @ A)     
    
    def satng(x, sat):
        if x < sat and x > -sat:
            return x
        elif x < -sat:
            return -sat
        else:
            return sat  
