import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from envPendulum import FurutaPendulum

class GraphSimulation():     
    def __init__(self):
        self.FurutaPendulum = FurutaPendulum()
            
    def Plot_phase_maps(self, controller, id_controller): 
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))            

        arm_angle_limits = (-np.pi, np.pi)
        arm_velocity_limits = (-10, 10)
        self._Fig_phase_map(axs[0], 0, 1, None, 'Ângulo do Braço [rad]', 'Velocidade Angular do Braço [rad/s]', 
                            controller, id_controller, arm_angle_limits, arm_velocity_limits)

        pendulum_angle_limits = (np.pi-0.4, np.pi+0.4)
        pendulum_velocity_limits = (-10, 10)

        equilibrium_points_instables = np.array([[np.pi, 0]])
        equilibrium_points_stables = np.array([])

        self._Fig_phase_map(axs[1], 2, 3, (equilibrium_points_instables, equilibrium_points_stables), 
                            'Ângulo do Pêndulo [rad]', 'Velocidade Angular do Pêndulo [rad/s]', controller, 
                            id_controller, pendulum_angle_limits, pendulum_velocity_limits)


        plt.tight_layout()
        plt.show()
            
    def _Fig_phase_map(self, ax, indexX, indexY, equilibriumPoints, xLabel, yLabel, controller, id_controller, x_limits, y_limits):
        X, Y, derivatesX, derivatesY, magnitude = self._Data_for_phase_map(indexX, indexY, controller, id_controller, x_limits, y_limits)

        if equilibriumPoints is not None:
            equilibriumPoints_instables, equilibriumPoints_stables = equilibriumPoints
            if equilibriumPoints_instables.size > 0:
                ax.scatter(equilibriumPoints_instables[:, 0], equilibriumPoints_instables[:, 1], color='r', s=10, zorder=2, label="Ponto de Equilíbrio Instável")
            if equilibriumPoints_stables.size > 0:
                ax.scatter(equilibriumPoints_stables[:, 0], equilibriumPoints_stables[:, 1], color='g', s=10, zorder=2, label="Ponto de Equilíbrio Estável")

        ax.streamplot(X, Y, derivatesX, derivatesY, color='b', linewidth=magnitude, density=1.6)
        #ax.set_title(f'{xLabel} x {yLabel}')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.grid()

        # Adiciona uma legenda se houver pontos de equilíbrio plotados
        if equilibriumPoints is not None and (equilibriumPoints_instables.size > 0 or equilibriumPoints_stables.size > 0):
            ax.legend(loc='upper right')
            
    def _Data_for_phase_map(self, indexX, indexY, controller, id_controller, x_limits, y_limits):
        numPoints = 100
        X, Y = np.meshgrid(np.linspace(x_limits[0], x_limits[1], numPoints), 
                        np.linspace(y_limits[0], y_limits[1], numPoints)) 

        derivatesX = np.zeros_like(X)
        derivatesY = np.zeros_like(Y)

        for i in range(numPoints):
            for j in range(numPoints):
                state = np.zeros(4)
                state[indexX] = X[i, j]     
                state[indexY] = Y[i, j]   

                # without control
                if id_controller == 0:  
                    derivates = self.FurutaPendulum.Dynamic(0, state, 0)

                # vlqr control    
                elif id_controller == 4:
                    derivates = self.FurutaPendulum.Dynamic(0, state, controller.signal_control(state, state*0.99))

                # pi, lqr, swing-up or rl control
                else:                    
                    derivates = self.FurutaPendulum.Dynamic(0, state, controller.signal_control(state))
                    
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