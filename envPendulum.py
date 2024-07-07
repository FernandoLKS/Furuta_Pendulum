import numpy as np

class PendulumFuruta():
   
    def __init__(self):
        self.Mb = 0.173
        self.Mp = 0.029
        self.r = 0.005
        self.Lp = 0.15
        self.lp = 0.075        
        self.Jb = 0.00094057
        self.Jp = 0.0002175
        self.Bb = 0.0008
        self.Bp = 0.0008
        self.tau_m = 0
        self.gravityForce = 9.81    
        
        self.armAngle = 0
        self.armVelocity = 0
        self.armAcceleration = 0
        self.pendulumAngle = 0
        self.pendulumVelocity = 0
        self.pendulumAcceleration = 0
      
    def InitialConditions(self):
        self.armAngle = np.pi
        self.armVelocity = 0
        self.pendulumAngle = 4/3 * np.pi
        self.pendulumVelocity = 0
        
        return [self.armAngle, self.armVelocity, self.pendulumAngle, self.pendulumVelocity]    
        
    def Dynamic(self, t, states, torque=0):
        self.armAngle = states[0]
        self.armVelocity = states[1]
        self.pendulumAngle = states[2] 
        self.pendulumVelocity = states[3]  
        
        self.tau_m = torque  
        
        self.armAcceleration = (self.tau_m - (self.Bb * self.armVelocity) + ((self.Mp * self.lp * self.r) * ((self.pendulumAcceleration * np.cos(self.pendulumAngle)) - (self.pendulumVelocity**2 * np.sin(self.pendulumAngle))))) / (self.Jb + (self.Mp * self.r**2))
        self.pendulumAcceleration = ((-self.Bb * self.pendulumVelocity) + (self.Mp * self.lp * self.r * self.armAcceleration * np.cos(self.pendulumAngle)) - (self.Mp * self.lp * self.gravityForce * np.sin(self.pendulumAngle))) / (self.Jp + (self.Mp * self.lp**2))
                  
        derivates = [self.armVelocity, self.armAcceleration, self.pendulumVelocity, self.pendulumAcceleration]
        return derivates          