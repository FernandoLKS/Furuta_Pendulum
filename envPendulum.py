import numpy as np

class FurutaPendulum():

    def __init__(self):
        self.Mb = 0.173                     # Mass of the arm (kg)
        self.Mp = 0.029                     # Mass of the pendulum (kg)
        self.r = 0.005                      # Radius of the arm (m)
        self.Lp = 0.15                      # Total length of the pendulum (m)
        self.lp = 0.075                     # Length to the pendulum's center of mass (m)
        self.Jb = 0.00094057                # Moment of inertia of the arm (kg.m^2)
        self.Jp = 0.0002175                 # Moment of inertia of the pendulum (kg.m^2)
        self.Bb = 0.0008                    # Viscous friction coefficient for the arm (N.m.s)
        self.Bp = 0.0008                    # Viscous friction coefficient for the pendulum (N.m.s)
        self.tau_m = 0                      # Applied torque (N.m)
        self.gravityForce = 9.81            # Gravitational acceleration (m/s^2) 
        
        self.armAngle = 0                   # Angle of the arm (rad)
        self.armVelocity = 0                # Velocity of the arm (rad/s)
        self.armAcceleration = 0            # Acceleration of the arm (rad/s^2)
        self.pendulumAngle = 0              # Angle of the pendulum (rad)
        self.pendulumVelocity = 0           # Velocity of the pendulum (rad/s)
        self.pendulumAcceleration = 0       # Acceleration of the pendulum (rad/s^2)
      
    def set_InitialConditions(self, armAngle=0, armVelocity=0, pendulumAngle=0, pendulumVelocity=0):
        self.armAngle = armAngle
        self.armVelocity = armVelocity
        self.pendulumAngle = pendulumAngle
        self.pendulumVelocity = pendulumVelocity
    
    def get_InitialConditions(self):
        return [self.armAngle, self.armVelocity, self.pendulumAngle, self.pendulumVelocity]    
    
    def Dynamic(self, t, state, torque=0):
        self.armAngle = state[0]
        self.armVelocity = state[1]
        self.pendulumAngle = state[2] 
        self.pendulumVelocity = state[3]  
        
        self.tau_m = torque  
        
        self.armAcceleration = (self.tau_m - (self.Bb * self.armVelocity) + ((self.Mp * self.lp * self.r) * ((self.pendulumAcceleration * np.cos(self.pendulumAngle)) - (self.pendulumVelocity**2 * np.sin(self.pendulumAngle))))) / (self.Jb + (self.Mp * self.r**2))
        self.pendulumAcceleration = ((-self.Bb * self.pendulumVelocity) + (self.Mp * self.lp * self.r * self.armAcceleration * np.cos(self.pendulumAngle)) - (self.Mp * self.lp * self.gravityForce * np.sin(self.pendulumAngle))) / (self.Jp + (self.Mp * self.lp**2))
                  
        derivates = [self.armVelocity, self.armAcceleration, self.pendulumVelocity, self.pendulumAcceleration]
        return derivates
