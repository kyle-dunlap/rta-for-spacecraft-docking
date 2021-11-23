"""
Author: Kyle Dunlap, University of Cincinnati, dunlapkp@mail.uc.edu
Containts Four different RTA classes
"""
import numpy as np
import scipy
from scipy import linalg
from scipy import integrate
import quadprog
import constraint


class Parameters():
    """
    Define parameters used for simulations
    """
    def __init__(self):
        self.n = 0.001027  # mean motion (rad/sec)
        self.m = 12  # mass (kg)
        self.dt = 1  # time step (sec)
        self.max_t = 5000  # max simulation time (sec)
        self.u_max = 1.  # max control input (N)
        self.nu0 = 0.2  # maximum allowable docking velocity (m/s)
        self.nu1 = 2.15*self.n  # distance dependent speed limit slope (m/s)
        self.max_vel = 10  # maximum allowable velocity (m/s)
        self.Rmax = 10000  # Maximum distance (m)

        # Verify Proofs:
        # Theorem III.2:
        if self.max_vel*(self.nu1*3**0.5+3*self.n**2*3**0.5/self.nu1+2*self.n)-3*self.n**2*self.nu0/self.nu1 > self.u_max/self.m:
            print('Theorem III.2 not valid')
        # Theorem III.3:
        if (3*self.n**2*self.Rmax+2*self.n*self.max_vel)**2 > (self.u_max/self.m)**2:
            print('Theorem III.3 not valid')
        if (-2*self.n*self.max_vel)**2 > (self.u_max/self.m)**2:
            print('Theorem III.3 not valid')
        if (-self.n**2*self.Rmax)**2 > (self.u_max/self.m)**2:
            print('Theorem III.3 not valid')

        # A matrix: x = [x, y, z, xdot, ydot, zdot]^T
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [3 * self.n ** 2, 0, 0, 0, 2 * self.n, 0],
                           [0, 0, 0, -2 * self.n, 0, 0],
                           [0, 0, -self.n ** 2, 0, 0, 0]], dtype=np.float64)

        # B matrix: u = [Fx, Fy, Fz]^T
        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1 / self.m, 0, 0],
                           [0, 1 / self.m, 0],
                           [0, 0, 1 / self.m]], dtype=np.float64)


class speed_limit(Parameters):
    """
    Distance dependent speed limit
    h_x >= 0 when constraint is satisfied
    """
    def h_x(self, x):
        return self.nu0 + self.nu1*np.linalg.norm(x[0:3]) - np.linalg.norm(x[3:6])

    # gradient of safety constraint
    def grad(self, x):
        Hs = np.array([[2*self.nu1**2, 0, 0, 0, 0, 0],
                       [0, 2*self.nu1**2, 0, 0, 0, 0],
                       [0, 0, 2*self.nu1**2, 0, 0, 0],
                       [0, 0, 0, -2, 0, 0],
                       [0, 0, 0, 0, -2, 0],
                       [0, 0, 0, 0, 0, -2]])

        ghs = Hs @ x
        ghs[0] = ghs[0] + 2*self.nu1*self.nu0*x[0]/np.linalg.norm(x[0:3])
        ghs[1] = ghs[1] + 2*self.nu1*self.nu0*x[1]/np.linalg.norm(x[0:3])
        ghs[2] = ghs[2] + 2*self.nu1*self.nu0*x[2]/np.linalg.norm(x[0:3])
        return ghs

    # Class-kappa strengthening function
    def alpha(self, x):
        return 0.05*x + 0.1*x**3


class xd_limit(Parameters):
    """
    maximum xdot constraint
    h_x >= 0 when constraint is satisfied
    """
    def h_x(self, x):
        return self.max_vel**2 - x[3]**2

    # gradient of safety constraint
    def grad(self, x):
        Hs = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -2, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])
        return Hs @ x

    # Class-kappa strengthening function
    def alpha(self, x):
        return 0.0005*x + 0.001*x**3


class yd_limit(Parameters):
    """
    maximum ydot constraint
    h_x >= 0 when constraint is satisfied
    """
    def h_x(self, x):
        return self.max_vel**2 - x[4]**2

    # gradient of safety constraint
    def grad(self, x):
        Hs = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -2, 0],
                       [0, 0, 0, 0, 0, 0]])
        return Hs @ x

    # Class-kappa strengthening function
    def alpha(self, x):
        return 0.0005*x + 0.001*x**3


class zd_limit(Parameters):
    """
    maximum zdot constraint
    h_x >= 0 when constraint is satisfied
    """
    def h_x(self, x):
        return self.max_vel**2 - x[5]**2

    # gradient of safety constraint
    def grad(self, x):
        Hs = np.array([[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -2]])
        return Hs @ x

    # Class-kappa strengthening function
    def alpha(self, x):
        return 0.0005*x + 0.001*x**3


class RTA(Parameters):
    """
    Setup parameters that apply to multiple RTAs
    """
    def __init__(self):
        super().__init__()

        # define list of constraints to be used
        self.speed_limit = speed_limit()
        self.xd_limit = xd_limit()
        self.yd_limit = yd_limit()
        self.zd_limit = zd_limit()
        self.constraint_list = ['speed_limit', 'xd_limit', 'yd_limit', 'zd_limit']

    # All common implicit RTA parameters
    def setup_implicit_parameters(self):
        self.T_backup = 5  # length of time in backup trajectory horizon (sec)
        self.Nsteps = int(self.T_backup/self.dt) + 1  # number of steps in horizon of backup trajectory
        self.Nskip = 1  # skip points when checking discrete trajectory points in optimization
        self.N_checkall = 5  # check first points along trajectory
        self.phi = np.zeros([6, self.Nsteps])  # tracks states along trajectory
        self.S = np.zeros([6, 6, self.Nsteps])  # Sensitivity matrix

        self.C = np.eye(6)  # C matrix for LQR tracking
        A_int = np.vstack((np.hstack((self.A, np.zeros((6, 6)))), np.hstack((self.C, np.zeros((6, 6))))))  # A matrix for LQR tracking
        B_int = np.vstack((self.B, np.zeros((6, 3))))  # B matrix for LQR tracking
        # LQR gain matrices
        Q = np.eye(12)*1e-5
        R = np.eye(3)*1e7
        # Solve the Algebraic Ricatti equation for the given system
        P = scipy.linalg.solve_continuous_are(A_int, B_int, Q, R)
        # Construct the constain gain matrix, K
        K = np.linalg.inv(R) @ (np.transpose(B_int) @ P)
        self.K_1 = K[:, 0:6]
        self.K_2 = K[:, 6:]

        self.z = np.zeros((6, 1))  # error between x and x_des
        self.eps = 1.5  # acceptable error
        self.track = False  # tracks NMT if True
        self.used_ub = False  # Determines if the RTA recently used the backup controller
        self.steps = 20  # Number of steps between updates to x_des

        # Find all NMTs that adhere to constraint
        problem = constraint.Problem()
        self.points = 20
        problem.addVariable('theta1', np.linspace(np.pi/10, np.pi/2-np.pi/10, self.points))
        problem.addVariable('theta2', np.linspace(0.001, np.pi, self.points))
        problem.addConstraint(self.safety_constraint, ['theta1', 'theta2'])
        self.solutions = problem.getSolutions()
        self.NMTs = []
        self.z_coefs = []
        # For all acceptable NMTs, save them to an array
        for index, solution in enumerate(self.solutions):
            theta1 = solution['theta1']
            theta2 = solution['theta2']
            z_coef = 1/np.sin(theta1)*np.sqrt(np.tan(theta2)**2+4*np.cos(theta1)**2)
            for i1 in range(self.points):
                self.z_coefs.append(z_coef)
                psi = i1/self.points*2*np.pi
                nu = np.arctan(2*np.cos(theta1)/np.tan(theta2))+psi
                x_NMT = np.array([np.sin(nu), 2*np.cos(nu), z_coef*np.sin(psi), self.n*np.cos(nu), -2*self.n*np.sin(nu), z_coef*self.n*np.cos(psi)])
                self.NMTs.append(x_NMT)
        self.NMTs = np.array(self.NMTs)

    # LQR tracking backup controller used for implicit RTA
    # Note: There are some infrequent situations where the backup controller
    # could cause the system to briefly enter an unsafe state if the distance
    # between x and x_des is large. The backup controller will ultimately guide
    # the system back to a safe state after a short time.
    def implicit_backup_control(self, x, integrate=False):
        # If used to integrate trajectory
        if integrate:
            # Track NMT if within acceptable error range
            if np.linalg.norm(x[0:3] - self.int_x_des[0:3]) <= self.eps:
                self.int_track = True
            # Track NMT
            if self.int_track:
                self.int_x_des = self.int_x_des + (self.A @ self.int_x_des)*self.dt
            # Calculate u
            u = self.K_1 @ (self.int_x_des - x) - self.K_2 @ self.int_z
            # Update error
            self.int_z = self.int_z + (x - self.int_x_des)*self.dt
        # If used by RTA filter
        else:
            # Track NMT if within acceptable error range
            if np.linalg.norm(x[0:3] - self.x_des[0:3]) <= self.eps:
                self.track = True
            # Track NMT
            if self.track:
                self.x_des = self.x_des + (self.A @ self.x_des)*self.dt
            # Calculate u
            u = self.K_1 @ (self.x_des - x) - self.K_2 @ self.z
            # Update error
            self.z = self.z + (x - self.x_des)*self.dt
        return np.clip(u, -self.u_max, self.u_max)

    # constraint for acceptable NMTs
    def safety_constraint(self, theta1, theta2):
        if (np.tan(theta2)**2+4*np.cos(theta1)**2)/np.sin(theta1)**2 <= (self.nu1/self.n)**2 - 4:
            return True

    # Find closest acceptable NMT to current position
    def find_NMT(self, x):
        # semi minor axis
        b = x[0]/(np.sin(np.arctan(2*x[0]/x[1])))
        if b > 4868.5:
            b = b/b*4868.5
        dist = []
        NMTs = []

        # Find distance to points on NMTs
        for i in range(len(self.z_coefs)):
            if b*self.z_coefs[i] <= 9737:
                x_NMT = self.NMTs[i]*b
                NMTs.append(x_NMT)
                dist.append(np.linalg.norm(np.reshape(x[0:3], 3)-x_NMT[0:3]))

        # Return NMT with minimum distance to current point
        return np.reshape(NMTs[np.argmin(dist)], (6, 1))


class Explicit_Switching(RTA):
    """
    Explicitly define safety (offline)
    Switch to backup controller if u_des is unsafe
    """
    def get_u(self, x, u_des):
        # Integrate one step under u_des
        x1 = x + (self.A @ x + self.B @ u_des)*self.dt

        udes_safe = True
        # Determine if next state is unsafe for any constraint
        for str in self.constraint_list:
            c = getattr(self, str)
            if c.h_x(x1) < 0:
                udes_safe = False
                break

        # If u_des is safe, pass u_des to plant
        if udes_safe:
            u = u_des
        # Otherwise, pass u_b to plant
        else:
            u = self.u_b(x, x1)

        return np.clip(u, -self.u_max, self.u_max)

    # Explicit backup controller: h_x = 0
    def u_b(self, x, x1):
        # Make h_x = 0 for xdot limit
        if self.xd_limit.h_x(x1) < 0:
            x1[3] = self.max_vel*np.sign(x1[3])
        # Make h_x = 0 for ydot limit
        if self.yd_limit.h_x(x1) < 0:
            x1[4] = self.max_vel*np.sign(x1[4])
        # Make h_x = 0 for zdot limit
        if self.zd_limit.h_x(x1) < 0:
            x1[5] = self.max_vel*np.sign(x1[5])
        # Make h_x = 0 for speed limit
        if self.speed_limit.h_x(x1) < 0:
            rH = np.linalg.norm(x1[0:3])
            vH = np.linalg.norm(x1[3:6])
            vH_max = self.nu1 * rH + self.nu0
            x1[3:6] = x1[3:6]/vH*vH_max

        # Use inverse dynamics to find u
        acc = (x1[3:6] - x[3:6]) / self.dt
        u = (acc[0:3]-self.A[3:6]@x1)*self.m

        return u


class Explicit_Optimization(RTA):
    """
    Explicitly define safety (offline)
    Optimize control
    """
    def get_u(self, x, u_des):
        # Objective function: Minimize ||M-q||^2
        M = np.eye(3)
        q = np.reshape(u_des, 3)
        # Subject to constraints: G*u <= h
        # Actuation constraints:
        G = [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
        h = [-self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max]

        # For each safety constraint
        for str in self.constraint_list:
            c = getattr(self, str)

            # Calculate barrier constraint
            g_temp = np.reshape(c.grad(x), 6) @ self.B
            h_temp = -np.reshape(c.grad(x), 6) @ self.A @ x - c.alpha(c.h_x(x))

            # G*u <= h
            G.append([g_temp[0], g_temp[1], g_temp[2]])
            h.append(h_temp[0])

        # Solve optimization program!
        u_act = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)[0]
        return np.reshape(u_act, (3, 1))


class Implicit_Switching(RTA):
    """
    Implicitly define safety (online)
    Switch to backup controller if u_des is unsafe
    """
    def __init__(self):
        super().__init__()
        self.setup_implicit_parameters()

    def get_u(self, x, u_des):
        self.steps += 1
        # Every 100 steps, update x_des (closest NMT)
        if self.steps >= 20:
            self.x_des = self.find_NMT(x)
            self.steps = -1
            self.track = False

        # Take one step under "u_des"
        x1 = x + (self.A @ x + self.B @ u_des)*self.dt
        # Integrate trajectory from x1
        self.integrate(x1)

        udes_safe = True
        # Determine if any state along trajectory is unsafe for any constraint
        for i in range(0, self.Nsteps):
            for str in self.constraint_list:
                c = getattr(self, str)
                if c.h_x(x1) < 0:
                    udes_safe = False
                    break

        # If u_des is safe, pass u_des to plant
        if udes_safe:
            u = u_des
            self.track = False
        # Otherwise, pass u_b to plant
        else:
            u = self.u_b(x)

        return np.clip(u, -self.u_max, self.u_max)

    # Use implicit backup controller
    def u_b(self, x, integrate=False):
        u = self.implicit_backup_control(x, integrate)
        return np.clip(u, -self.u_max, self.u_max)

    # Integrate trajectory
    def integrate(self, x):
        # Set initial parameters
        self.phi[:, 0] = np.reshape(x, 6)
        self.int_x_des = self.x_des
        self.int_track = self.track
        self.int_z = self.z

        # For next steps
        for i in range(1, self.Nsteps):
            # Use dynamics to take step with backup controller
            x = np.reshape(self.phi[:, i-1], (6, 1))
            u_des = self.u_b(x, integrate=True)
            x1 = x + (self.A @ x + self.B @ u_des)*self.dt
            # Save state to array
            self.phi[:, i] = np.reshape(x1, 6)


class Implicit_Optimization(RTA):
    """
    Implicitly define safety (online)
    Optimize control
    """
    def __init__(self):
        super().__init__()
        self.setup_implicit_parameters()

    def get_u(self, x, u_des):
        self.steps += 1
        # Every 100 steps, update x_des (closest NMT)
        if self.steps >= 20:
            self.x_des = np.reshape(self.find_NMT(x), (6, 1))
            self.steps = -1
            self.track = False

        # Integrate trajectory
        self.integrate(x)

        # Objective function: Minimize ||M-q||^2
        M = np.eye(3)
        q = np.reshape(u_des, 3)
        # Subject to constraints: G*u <= h
        # Actuation constraints:
        self.G = [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
        self.h = [-self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max]

        # save for use in invariance constraints
        self.Ax0 = self.A @ x

        # For first 'N_checkall' states, append invariance constraints
        for i in range(0, self.N_checkall):
            self.invariance_constraints(i)

        # For remaining states, append invariance constraints
        for i in range(self.N_checkall+self.Nskip, self.Nsteps, self.Nskip):
            self.invariance_constraints(i)

        # Solve optimization program!
        u_act = quadprog.solve_qp(M, q, np.array(self.G).T, np.array(self.h), 0)[0]
        u = np.reshape(u_act, (3, 1))

        return np.clip(u, -self.u_max, self.u_max)

    def invariance_constraints(self, i):
        # For each safety constraint
        for str in self.constraint_list:
            c = getattr(self, str)

            # Construct barrier constraint
            g_temp = c.grad(self.phi[:, i]) @ (self.S[:, :, i] @ self.B)
            h_temp = c.grad(self.phi[:, i]) @ (self.S[:, :, i] @ self.Ax0) + c.alpha(c.h_x(self.phi[:, i]))

            # G*u <= h
            self.G.append([g_temp[0], g_temp[1], g_temp[2]])
            self.h.append(-h_temp[0])

    # Use implicit backup controller
    def u_b(self, x, integrate=False):
        u = self.implicit_backup_control(x, integrate)
        return np.clip(u, -self.u_max, self.u_max)

    # Integrate trajectory
    def integrate(self, x):
        # Set initial parameters
        self.phi[:, 0] = np.reshape(x, 6)
        self.S[:, :, 0] = np.eye(6)
        self.int_x_des = self.x_des
        self.int_track = self.track
        self.int_z = self.z

        # For next steps
        for i in range(1, self.Nsteps):
            # Use dynamics to take step with backup controller
            x = np.reshape(self.phi[:, i-1], (6, 1))
            u_des = self.u_b(x, integrate=True)
            x1 = x + (self.A @ x + self.B @ u_des)*self.dt
            # Save state to array
            self.phi[:, i] = np.reshape(x1, 6)
            # Update sensitivity matrix
            Dphi = self.get_Jacobian(self.phi[:, i])
            self.S[:, :, i] = self.S[:, :, i-1] + (Dphi @ self.S[:, :, i-1])*self.dt

    # Calculate Jacobian
    def get_Jacobian(self, phi):
        return self.A - self.B @ self.K_1
