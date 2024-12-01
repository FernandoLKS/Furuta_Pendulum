import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from envPendulum import FurutaPendulum

class GraphDynamicSimulation():     
    def __init__(self):
        self.FurutaPendulum = FurutaPendulum()

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