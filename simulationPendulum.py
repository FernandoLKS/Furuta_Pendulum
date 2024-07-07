from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Simulation():  
    
    def __init__(self, FurutaPendulum, ts, max_iterations, torque):
        self.ts = ts
        self.max_iterations = max_iterations
        self.torque = torque
        
        self.states = None
        self.FurutaPendulum = FurutaPendulum    
        
        self.Execute()
         
    def Execute(self):  
        self.states = np.zeros((self.max_iterations, 4))   
        self.states[0, :] = self.FurutaPendulum.InitialConditions()               
        
        for i in range(1, self.max_iterations):
            solution = solve_ivp(lambda t, states: self.FurutaPendulum.Dynamic(t, states, self.torque), [0, self.ts], self.states[i-1, :], method='RK45')
            self.states[i, :] = solution.y[:, -1]   
        
        self.PlotPhaseMaps()       
        self.PlotSimulation()     
        
    def PlotPhaseMaps(self):

        def derivateVectors(ax, indexX, indexY, equilibriumPoints, xLabel, yLabel):
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
                    derivates = self.FurutaPendulum.Dynamic(0, state, 0)

                    derivatesX[i, j] = derivates[indexX]
                    derivatesY[i, j] = derivates[indexY]

            magnitude = np.sqrt(derivatesX**2 + derivatesY**2)
            magnitude /= np.max(magnitude)

            if equilibriumPoints is not None:
                ax.scatter(equilibriumPoints[:, 0], equilibriumPoints[:, 1], color='r', s=10, zorder=2)           

            ax.streamplot(X, Y, derivatesX, derivatesY, color='b', linewidth=magnitude, density=1.6)
            ax.set_title(f'{xLabel} x {yLabel}')
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.grid()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Arm angle x arm velocity      
        derivateVectors(axs[0], 0, 1, None, 'Arm angle [rad]', 'Arm velocity [rad/s]')        

        # Pendulum angle x pendulum velocity  
        equilibriumPointsPendulum = np.array([[-np.pi, 0], [0,0], [np.pi,0]])
        derivateVectors(axs[1], 2, 3, equilibriumPointsPendulum, 'Pendulum angle [rad]', 'Pendulum velocity [rad/s]')

        plt.tight_layout()
        plt.show()

    def PlotSimulation(self):
        def setupAxis(ax, title, limit):            
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

        def animate(i, xData, yData, line, trace, timeTemplate, timeText):
            thisX = [0, xData[i]]
            thisY = [0, yData[i]]
            historyX = xData[:i]
            historyY = yData[:i]
            line.set_data(thisX, thisY)
            trace.set_data(historyX, historyY)
            timeText.set_text(timeTemplate % (i * self.ts))
            return line, trace, timeText

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Arm plot setup
        xDataArm = self.FurutaPendulum.r * np.sin(self.states[:, 0])
        yDataArm = -self.FurutaPendulum.r * np.cos(self.states[:, 0])
        lineArm, traceArm, timeTemplateArm, timeTextArm = setupAxis(axs[0], 'Arm Motion', self.FurutaPendulum.r)

        # Pendulum plot setup
        xDataPendulum = self.FurutaPendulum.Lp * np.sin(self.states[:, 2])
        yDataPendulum = -self.FurutaPendulum.Lp * np.cos(self.states[:, 2])
        linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum = setupAxis(axs[1], 'Pendulum Motion', self.FurutaPendulum.Lp)

        animationArm = animation.FuncAnimation(fig, animate, fargs=(xDataArm, yDataArm, lineArm, traceArm, timeTemplateArm, timeTextArm), frames=len(xDataArm), interval=10, blit=True)
        animationPendulum = animation.FuncAnimation(fig, animate, fargs=(xDataPendulum, yDataPendulum, linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum), frames=len(xDataPendulum), interval=10, blit=True)

        plt.tight_layout()
        plt.show()       