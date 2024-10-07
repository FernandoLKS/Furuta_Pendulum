import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from envPendulum import FurutaPendulum

class GraphSimulation(): 
    
    def __init__(self):
        self.FurutaPendulum = FurutaPendulum()
        
    def Plot_phase_maps(self): 
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))            
             
        self._Fig_phase_map(axs[0], 0, 1, None, 'Arm angle [rad]', 'Arm velocity [rad/s]')                                                  
        self._Fig_phase_map(axs[1], 2, 3, np.array([[-np.pi, 0], [0,0], [np.pi,0]]), 'Pendulum angle [rad]', 'Pendulum velocity [rad/s]')  

        plt.tight_layout()
        plt.show()
        
    def _Fig_phase_map(self, ax, indexX, indexY, equilibriumPoints, xLabel, yLabel):
        X, Y, derivatesX, derivatesY, magnitude = self._Data_for_phase_map(indexX, indexY)

        if equilibriumPoints is not None:
            ax.scatter(equilibriumPoints[:, 0], equilibriumPoints[:, 1], color='r', s=10, zorder=2)           

        ax.streamplot(X, Y, derivatesX, derivatesY, color='b', linewidth=magnitude, density=1.6)
        ax.set_title(f'{xLabel} x {yLabel}')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.grid()

    def _Data_for_phase_map(self, indexX, indexY):
        limit = 2 * np.pi 
        numPoints = 100
        X, Y = np.meshgrid(np.linspace(-limit, limit, numPoints), np.linspace(-limit, limit, numPoints)) 

        derivatesX = np.zeros_like(X)
        derivatesY = np.zeros_like(Y)

        for i in range(numPoints):
            for j in range(numPoints):
                state = np.zeros(4)
                state[indexX] = X[i, j]     
                state[indexY] = Y[i, j]                
                derivates = self.FurutaPendulum.Dynamic(0, state, self.FurutaPendulum.tau_m)

                derivatesX[i, j] = derivates[indexX]
                derivatesY[i, j] = derivates[indexY]

        magnitude = np.sqrt(derivatesX**2 + derivatesY**2)
        magnitude /= np.max(magnitude)
        
        return X, Y, derivatesX, derivatesY, magnitude

    def Plot_pendulum_simulation(self, states, ts):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        ArmAngles = states[:, 0]
        PendulumAngles = states[:, 2] 
        
        xDataArm, yDataArm = self._Data_for_pendulum_simulation(ArmAngles, self.FurutaPendulum.r)
        xDataPendulum, yDataPendulum = self._Data_for_pendulum_simulation(PendulumAngles, self.FurutaPendulum.Lp)      
   
        lineArm, traceArm, timeTemplateArm, timeTextArm = self._Fig_pendulum_simulation(axs[0], 'Arm Motion', self.FurutaPendulum.r)        
        linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum = self._Fig_pendulum_simulation(axs[1], 'Pendulum Motion', self.FurutaPendulum.Lp)

        arm_anim = animation.FuncAnimation(fig, self._Animate_pendulum_simulation, fargs=(xDataArm, yDataArm, lineArm, traceArm, timeTemplateArm, timeTextArm, ts), frames=len(xDataArm), interval=10, blit=True)
        pendulum_anim = animation.FuncAnimation(fig, self._Animate_pendulum_simulation, fargs=(xDataPendulum, yDataPendulum, linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum, ts), frames=len(xDataPendulum), interval=10, blit=True)

        plt.tight_layout()
        plt.show()
               
    def _Fig_pendulum_simulation(self, ax, title, limit):
        ax.set_title(title)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        trace, = ax.plot([], [], '.-', lw=1, ms=2)
        timeTemplate = 'time = %.1fs'
        timeText = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        return line, trace, timeTemplate, timeText
        
    def _Data_for_pendulum_simulation(self, states, length):      
        xData = length * np.sin(states)
        yData = -length * np.cos(states)

        return xData, yData   
    
    def _Animate_pendulum_simulation(self, i, xData, yData, line, trace, timeTemplate, timeText, ts):
        line.set_data([0, xData[i]], [0, yData[i]])
        trace.set_data(xData[:i], yData[:i])
        timeText.set_text(timeTemplate % (i * ts))
        return line, trace, timeText 