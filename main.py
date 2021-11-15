"""
Author: Kyle Dunlap, University of Cincinnati, dunlapkp@mail.uc.edu
Used to compare four different RTA classes for spacecraft docking
Run env.sim('classes') on line 204
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from rta import Explicit_Switching, Implicit_Switching, Explicit_Optimization, Implicit_Optimization, Parameters
plt.rcParams.update({"text.usetex": True, 'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amssymb}']})
plt.rcParams.update({'font.size': 18, 'figure.autolayout': True})


class DockingEnv(Parameters):
    """
    Used to simulate docking environment.
    """
    def __init__(self):
        super().__init__()
        # Define LQR Gains
        Q = np.eye(6) * 0.05
        R = np.eye(3) * 1000
        # Find K matrix
        Xare = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        K = np.matrix(scipy.linalg.inv(R)*(self.B.T*Xare))
        self.K = np.squeeze(np.asarray(K))

    # Resets environment to initial conditions
    def reset(self, random=True):
        self.time = 0
        # Randomize initial conditions
        if random:
            self.x = 2*np.random.rand(6, 1)-1
            self.x[0] = self.x[0]*5000
            self.x[1] = self.x[1]*5000
            self.x[2] = self.x[2]*5000
        # Standard initial conditions
        else:
            self.x = np.array([[-5000], [6000], [6000], [0.5], [-0.5], [0.5]])

        return self.x, False

    # Integrate one time step in environment
    def step(self, act):
        self.time += self.dt
        # Euler integration
        self.x = self.x + (self.A @ self.x + self.B @ act)*self.dt
        # Done if docked (||rH|| <= 1) or time limit has been reached
        if np.linalg.norm(np.reshape(self.x, 6)[0:3]) <= 1 or self.time >= self.max_t:
            done = True
        else:
            done = False

        return self.x, done

    # LQR primary controller
    def LQR(self):
        u = -self.K @ self.x
        u = np.clip(u, -self.u_max, self.u_max)
        return u

    # Run one simulation using each RTA in 'classes'
    def sim(self, classes, random=True):
        # Initialize plotting parameters
        rh_max = []
        vh_max = []
        t_max = []
        xd_max = []
        yd_max = []
        zd_max = []
        xd_min = []
        yd_min = []
        zd_min = []

        # Initialize state
        x0, _ = self.reset(random=random)

        for cls in classes:
            # Define plotting parameters
            ls = '-'
            lw = 2
            if cls == 'No RTA':
                clr = 'y'
                lbl = 'No RTA'
            if cls == 'Explicit_Switching':
                clr = 'b'
                lbl = 'Ex. Switch.'
            if cls == 'Explicit_Optimization':
                clr = 'r'
                lbl = 'Ex. Opt.'
            if cls == 'Implicit_Switching':
                clr = 'tab:green'
                lbl = 'Im. Switch.'
            if cls == 'Implicit_Optimization':
                clr = 'm'
                ls = ':'
                lw = 4
                lbl = 'Im. Opt.'

            # Import RTA class
            if cls != 'No RTA':
                rta = eval(cls+"()")

            # Reset environment
            _, done = self.reset(random=random)
            self.x = x0
            x = x0
            s = [np.reshape(x, 6)]
            us = [[0, 0, 0]]

            # Run one simulation
            while not done:
                # Primary controller
                u = self.LQR()
                # Get safe action from RTA
                if cls != 'No RTA':
                    u = rta.get_u(x, u)
                # Take step with action
                x, done = self.step(u)
                s = np.append(s, [np.reshape(x, 6)], axis=0)
                us = np.append(us, [np.reshape(u, 3)], axis=0)

            # Calculate relative position and relative velocity
            v = np.empty([len(s), 2])
            for i in range(len(s)):
                v[i, :] = [np.linalg.norm(s[i, 0:3])/1000, np.linalg.norm(s[i, 3:6])]

            # Find max of each value (for plotting)
            rh_max.append(np.max(v[:,0])*1.1)
            vh_max.append(np.max(v[:,1])*1.1)
            t_max.append(len(s)*1.1)
            xd_max.append(np.max(s[:,3])*1.1)
            yd_max.append(np.max(s[:,4])*1.1)
            zd_max.append(np.max(s[:,5])*1.1)
            xd_min.append(np.min(s[:,3])*1.1)
            yd_min.append(np.min(s[:,4])*1.1)
            zd_min.append(np.min(s[:,5])*1.1)

            # Plot data from simulation
            plt.figure(1)
            plt.plot(v[:, 0], v[:, 1], color=clr, linewidth=lw, linestyle=ls, label=lbl)
            plt.figure(2)
            plt.plot(range(len(s)), s[:, 3], color=clr, linewidth=lw, linestyle=ls, label=lbl)
            plt.figure(3)
            plt.plot(range(len(s)), s[:, 4], color=clr, linewidth=lw, linestyle=ls, label=lbl)
            plt.figure(4)
            plt.plot(range(len(s)), s[:, 5], color=clr, linewidth=lw, linestyle=ls, label=lbl)

        # Label plots
        plt.figure(1)
        plt.fill_between([0, 15], 0, [self.nu0, self.nu0+self.nu1*15000], color=(244/255, 249/255, 241/255))
        plt.fill_between([0, 15], [self.nu0, self.nu0+self.nu1*15000], [1000, 1000], color=(255/255, 239/255, 239/255))
        plt.plot([0, 15], [self.nu0, self.nu0+self.nu1*15000], 'k--', linewidth=2)
        plt.xlim([0, max(rh_max)])
        plt.ylim([0, max(vh_max)])
        plt.xlabel('Relative Position ($\Vert \pmb{r}_{\mathrm{H}} \Vert$) [km]')
        plt.ylabel('Relative Velocity ($\Vert \pmb{v}_{\mathrm{H}} \Vert$) [m/s]')
        plt.grid(True)

        plt.figure(2)
        plt.fill_between([0, 10000], [self.max_vel, self.max_vel], [100, 100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [-100, -100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [self.max_vel, self.max_vel], color=(244/255, 249/255, 241/255))
        plt.plot([0, max(t_max)], [self.max_vel, self.max_vel], 'k--', linewidth=2)
        plt.plot([0, max(t_max)], [-self.max_vel, -self.max_vel], 'k--', linewidth=2)
        plt.xlim([0, max(t_max)])
        plt.ylim([min(xd_min), max(xd_max)])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)

        plt.figure(3)
        plt.fill_between([0, 10000], [self.max_vel, self.max_vel], [100, 100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [-100, -100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [self.max_vel, self.max_vel], color=(244/255, 249/255, 241/255))
        plt.plot([0, max(t_max)], [self.max_vel, self.max_vel], 'k--', linewidth=2)
        plt.plot([0, max(t_max)], [-self.max_vel, -self.max_vel], 'k--', linewidth=2)
        plt.xlim([0, max(t_max)])
        plt.ylim([min(yd_min), max(yd_max)])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)

        plt.figure(4)
        plt.fill_between([0, 10000], [self.max_vel, self.max_vel], [100, 100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [-100, -100], color=(255/255, 239/255, 239/255))
        plt.fill_between([0, 10000], [-self.max_vel, -self.max_vel], [self.max_vel, self.max_vel], color=(244/255, 249/255, 241/255))
        plt.plot([0, max(t_max)], [self.max_vel, self.max_vel], 'k--', linewidth=2)
        plt.plot([0, max(t_max)], [-self.max_vel, -self.max_vel], 'k--', linewidth=2)
        plt.xlim([0, max(t_max)])
        plt.ylim([min(zd_min), max(zd_max)])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)
        plt.legend(facecolor='white', framealpha=1)

        plt.show()


# Setup environment
env = DockingEnv()
# Run one simulation with each RTA listed
env.sim(['No RTA','Explicit_Switching','Explicit_Optimization','Implicit_Switching','Implicit_Optimization'], random= False)
