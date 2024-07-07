import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class GraphSimulation():  
    '''
    Generate graphics and animations for the Furuta pendulum simulation.

    Inputs:
        simulation: Object 'Simulation'
    '''
    def __init__(self, simulation):
        self.simulation = simulation
        self.FurutaPendulum = simulation.get_FurutaPendulumInstance()
        self.states = simulation.get_States()
        
    def PlotPhaseMaps(self): 
        '''
        Show system phase maps.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))            
             
        self._derivateVectors(axs[0], 0, 1, None, 'Arm angle [rad]', 'Arm velocity [rad/s]')                                                 # Arm angle x arm velocity 
        self._derivateVectors(axs[1], 2, 3, np.array([[-np.pi, 0], [0,0], [np.pi,0]]), 'Pendulum angle [rad]', 'Pendulum velocity [rad/s]')  # Pendulum angle x pendulum velocity

        plt.tight_layout()
        plt.show()

    def PlotSimulation(self):
        '''
        Show system simulation.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        ArmAngles = self.states[:, 0]
        PendulumAngles = self.states[:, 2]        
        
        xDataArm = self.FurutaPendulum.r * np.sin(ArmAngles)
        yDataArm = -self.FurutaPendulum.r * np.cos(ArmAngles)
        lineArm, traceArm, timeTemplateArm, timeTextArm = self._setupAxis(axs[0], 'Arm Motion', self.FurutaPendulum.r)                           # Arm plot setup
        
        xDataPendulum = self.FurutaPendulum.Lp * np.sin(PendulumAngles)
        yDataPendulum = -self.FurutaPendulum.Lp * np.cos(PendulumAngles)
        linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum = self._setupAxis(axs[1], 'Pendulum Motion', self.FurutaPendulum.Lp) # Pendulum plot setup

        animationArm = animation.FuncAnimation(fig, self._animate, fargs=(xDataArm, yDataArm, lineArm, traceArm, timeTemplateArm, timeTextArm), frames=len(xDataArm), interval=10, blit=True)
        animationPendulum = animation.FuncAnimation(fig, self._animate, fargs=(xDataPendulum, yDataPendulum, linePendulum, tracePendulum, timeTemplatePendulum, timeTextPendulum), frames=len(xDataPendulum), interval=10, blit=True)

        plt.tight_layout()
        plt.show()       
           
    def _setupAxis(self, ax, title, limit):            
        '''
        Set up the axis for animation plot.

        Inputs:
            ax: Matplotlib axis object for plotting
            title: Title of the plot
            limit: Limit for the x and y axis
        
        Outputs:
            line: Line2D object for the main plot  
            trace: Line2D object for the trace plot
            timeTemplate: Template string for displaying time
            timeText: Text object for displaying time
        '''
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

    def _animate(self, i, xData, yData, line, trace, timeTemplate, timeText):
        '''
        Update the animation frame.

        Inputs:
            i: Current frame index
            xData: Array of x coordinates for the plot
            yData: Array of y coordinates for the plot
            line: Line2D object for the main plot
            trace: Line2D object for the trace plot
            timeTemplate: Template string for displaying time
            timeText: Text object for displaying time
        
        Outputs:
            line: Updated Line2D object for the main plot  
            trace: Updated Line2D object for the trace plot
            timeText: Updated Text object for displaying time
        '''
        thisX = [0, xData[i]]
        thisY = [0, yData[i]]
        historyX = xData[:i]
        historyY = yData[:i]
        line.set_data(thisX, thisY)
        trace.set_data(historyX, historyY)
        timeText.set_text(timeTemplate % (i * self.simulation.ts))
        return line, trace, timeText 
            
    def _derivateVectors(self, ax, indexX, indexY, equilibriumPoints, xLabel, yLabel):
        '''
        Calculate ODEs for map points.

        Inputs:
            ax: Matplotlib axis object for plotting
            indexX: Index of the state variable to be plotted on the x-axis
            indexY: Index of the state variable to be plotted on the y-axis
            equilibriumPoints: Array of equilibrium points to be plotted (optional)
            xLabel: Label for the x-axis
            yLabel: Label for the y-axis
        '''
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

        if equilibriumPoints is not None:
            ax.scatter(equilibriumPoints[:, 0], equilibriumPoints[:, 1], color='r', s=10, zorder=2)           

        ax.streamplot(X, Y, derivatesX, derivatesY, color='b', linewidth=magnitude, density=1.6)
        ax.set_title(f'{xLabel} x {yLabel}')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.grid()