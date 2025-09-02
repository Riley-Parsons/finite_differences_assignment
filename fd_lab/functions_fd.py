import numpy as np
import matplotlib.pyplot as plt
import math
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


class SolverPoissonXY(object):
    """
    Class containing attributes and methods for solving the Poisson PDE in two-dimensional Cartesian coordinates.

    Attributes:
        nx (int): number of mesh points along the x dimension.
        ny (int): number of mesh points along the y dimension.
        n (int): total number of mesh points.
        x (1D array): mesh coordinates along the x dimension.
        y (1D array): mesh coordinates along the y dimension.
        dx (float): mesh spacing along the x dimension.
        dy (float): mesh spacing along the y dimension.
        bc_x0 (dict): dictionary storing information for left boundary.
        bc_x1 (dict): dictionary storing information for right boundary.
        bc_y0 (dict): dictionary storing information for bottom boundary.
        bc_y1 (dict): dictionary storing information for top boundary.
        poisson_function (callable): Poisson function.
        a (2D array): coefficient matrix in system of equations to solve for PDE solution.
        b (1D array): vector of constants in system of equations to solve for PDE solution.
        solution (2D array): PDE solution array on the mesh.

    Arguments:
        xlim (1D array): lower and upper limits in x dimension.
        ylim (1D array): lower and upper limits in y dimension.
        delta (float): desired mesh spacing in x and y dimension i.e. assume uniform mesh spacing
        bc_x0 (dict): dictionary storing information for left boundary u(x0,y).
        bc_x1 (dict): dictionary storing information for right boundary u(x1,y).
        bc_y0 (dict): dictionary storing information for bottom boundary u(x,y0).
        bc_y1 (dict): dictionary storing information for top boundary u(x,y1).
        poisson_function (callable): Poisson function.
    """

    def __init__(self, xlim, ylim, delta, bc_x0, bc_x1, bc_y0, bc_y1, poisson_function):

        # TODO: define the integer number of mesh points (nx, ny, n), including boundaries, based on desired mesh spacing
        #nearest number, can be courser OR finer
        self.nx = int(round((xlim[1] - xlim[0])/delta +1))
        self.ny = int(round((ylim[1]-ylim[0])/delta+1))
        self.n = self.ny * self.nx
        logger.info("x points: %s", self.nx)
        logger.info("y points: %s", self.ny)
        logger.info("total points: %s", self.n)
        
        # TODO: calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.y = np.linspace(ylim[0], ylim[1], self.ny)
        
        # TODO: calculate the actual mesh spacing in x and y (dx, dy), may differ slightly from delta if not exactly divisible
        # assumes nx and ny >1
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        logger.info("delta x: %s", self.dx)
        logger.info("delta y: %s", self.dy)
        if not np.isclose(self.dx, self.dy):
            logger.warning("delta y and delta x not close!")
        
        # TODO: initialise linear algebra matrices (a,b)
        self.a = np.zeros((self.n,self.n))
        self.b = np.zeros(self.n)
        
        # store the four boundary conditions
        self.bc_x0 = bc_x0 #left
        self.bc_x1 = bc_x1 #right
        self.bc_y0 = bc_y0 #bottom
        self.bc_y1 = bc_y1 #top
        logger.debug("x0 condition: %s", self.bc_x0['type'])
        logger.debug("x1 condition: %s", self.bc_x1['type'])
        logger.debug("y0 condition: %s", self.bc_y0['type'])
        logger.debug("y1 condition: %s", self.bc_y1['type'])

        # equation corresponding to forcing function
        self.poisson_function = poisson_function

        # create solution matrix attribute
        self.solution = None
        
        # converts a list of floats to a list of ints
        def toint(lis):
            return [int(val) for val in lis]
                
        # find indexes for boundary conditions (K = ...)
        self.Bind = list(range(0, self.nx))
        self.Tind = list(range(self.n-self.nx, self.n))
        Lind = np.linspace(0, self.n-self.nx, self.ny)
        self.Lind = toint(Lind.tolist())
        Rind = np.linspace(self.nx-1, self.n-1, self.ny)
        self.Rind = toint(Rind.tolist())
        
        # set of K indexes that are boundary points
        self.boundary_i = set(self.Bind+self.Tind+self.Lind+self.Rind) 
        logger.debug("boundary indexes: %s", self.boundary_i)

    def bound_indexer(self, index):
        """
        turns boundary indexes to i and j
        """
        i = index%self.nx # i index
        j = index//self.nx # j index
        
        return i, j
        
        

    def dirichlet(self):
        """
        Apply Dirichlet boundary conditions to update the corresponding elements of the A matrix and b vector for the
        mesh points along the Dirichlet boundaries.
        """
        # TODO - your code here
        logger.info("Running dirchlet")
        for k in self.boundary_i:

            
            # checks if dirichlet and performs appropriate boundary func for index
            # bottom row
            if k in self.Bind and self.bc_y0['type'] == 'dirichlet':
                self.a[k,:] = 0.0
                self.a[k,k] = 1.0
                self.b[k] = self.bc_y0["function"](self.x[k%self.nx], self.y[0])
                
            #top row
            elif k in self.Tind and self.bc_y1['type'] == 'dirichlet':
                self.a[k,:] = 0.0
                self.a[k,k] = 1.0
                self.b[k] = self.bc_y1['function'](self.x[k%self.nx], self.y[self.ny-1])
            
            # left column
            elif k in self.Lind and self.bc_x0['type'] == 'dirichlet':
                self.a[k,:] = 0.0
                self.a[k,k] = 1.0
                self.b[k] = self.bc_x0['function'](self.x[0], self.y[k//self.nx])
            
            # right column
            elif k in self.Rind and self.bc_x1['type'] == 'dirichlet':
                self.a[k,:] = 0.0
                self.a[k,k] = 1.0
                self.b[k] = self.bc_x1['function'](self.x[self.nx-1], self.y[k//self.nx])
                

                
        logger.debug("b vector:\n%s", np.array2string(self.b, precision=3, suppress_small=True))
        logger.debug("A matrix:\n%s", np.array2string(self.a, precision=1, suppress_small=True, max_line_width=120))




    def neumann(self):
        """
        Apply Neumann boundary conditions to update the corresponding elements of the A matrix and b vector for the
        mesh points along the Neumann boundaries.
        """
        # TODO - your code here (Not needed for task 1A)
        logger.info("Running neumann")
        for k in self.boundary_i:
            i, j = self.bound_indexer(k)

            #top row
            if j == self.ny-1 and (i!= 0 and i != self.nx-1) and self.bc_y1["type"] == "neumann":
                self.a[k,:] = 0.0
                self.a[k, k-1] = 1.0
                self.a[k, k+1]=1.0
                self.a[k, k] = -4.0
                self.a[k, k-self.nx] = 2.0
                
                self.b[k] = (self.dy**2) * self.poisson_function(self.x[i], self.y[j]) - 2* self.dy * self.bc_y1['function'](self.x[i], self.y[j])
            
            # bottom row
            elif j == 0 and (i!=0 and i != self.nx-1) and self.bc_y0['type'] == "neumann":
                self.a[k, :] = 0.0
                self.a[k, k] = -4.0
                self.a[k, k-1] = 1.0
                self.a[k, k+1] = 1.0
                self.a[k, k+self.nx] = 2.0
            
                self.b[k] = (self.dy**2) * self.poisson_function(self.x[i], self.y[j]) + 2* self.dy * self.bc_y0['function'](self.x[i], self.y[j])
            # left column  
            elif i == 0 and (j != 0 and j != self.ny-1) and self.bc_x0['type'] == 'neumann':
                self.a[k,:] = 0.0
                self.a[k,k] = -4.0
                self.a[k,k+1] = 2
                self.a[k, k-self.nx] = 1
                self.a[k,k+self.nx] = 1
                
                self.b[k] = (self.dx**2) * self.poisson_function(self.x[i], self.y[j]) + 2* self.dx * self.bc_x0['function'](self.x[i], self.y[j])            
            # right column    
            elif i == self.nx-1 and j!=0 and j!= self.ny-1 and self.bc_x1['type'] == 'neumann':
                self.a[k, :] = 0.0
                self.a[k,k] = -4
                self.a[k, k-1] = 2.0
                self.a[k, k+self.nx] = 1
                self.a[k, k-self.nx] = 1
            
                self.b[k] = (self.dx**2) * self.poisson_function(self.x[i], self.y[j]) - 2* self.dx * self.bc_x1['function'](self.x[i], self.y[j])       

        logger.debug("b vector:\n%s", np.array2string(self.b, precision=3, suppress_small=True))
        logger.debug("A matrix:\n%s", np.array2string(self.a, precision=1, suppress_small=True, max_line_width=120))
            
            
        
     
    def internal(self):
        """
        Apply FD stencil and Poisson equation to update the corresponding elements of the A matrix and b vector for the
        internal mesh points.
        """
        # TODO - your code here
        logger.info("Running internal")
        n = set(range(self.n))
        internal_i = n - self.boundary_i # internal point ind (K)
        logger.debug("Internal boundary points: %s", internal_i)
        
        # assign stencil values at internal rows
        def stenciler(matrix, kval):
            
            matrix[kval, :] = 0.0
            matrix[kval, kval] = -4.0
            
            # check for wraparound
            if (kval%self.nx < 1 or kval%self.nx > self.nx-2):
                logger.critical("kval i index out of range")
            
            if (kval//self.nx < 1 or kval//self.nx > self.ny-2):
                logger.critical("kval j index out of range")
            
            for n in [kval-1, kval+1, kval-self.nx, kval+self.nx]:
                matrix[kval, n] = 1
                logger.debug("matrix i val = %s, j val = %s", n%self.nx, n//self.nx)
        

        
        for k in internal_i:
            stenciler(self.a, k)
            self.b[k] = (self.dx**2)*self.poisson_function(self.x[k%self.nx], self.y[k//self.nx])
            logger.debug("Internal b value: %s", self.b[k])
        
        
        logger.debug("internal A matrix:\n%s", np.array2string(self.a, precision =1, suppress_small = True, max_line_width = 120))
        



    def solve(self):
        """
        Update A and b in the system of equations, Au=b, then solve the system for u and re-shape to store on
        the mesh as the numerical solution.
        """

        # TODO - your code here
        logger.info("Running solve")
        self.neumann()
        self.internal()
        self.dirichlet()
        u = np.linalg.solve(self.a, self.b)
        
        logger.debug("U matrix: %s", np.array2string(u, precision = 3, suppress_small = True))
        
        n = set(range(self.n))
        internal_i = n - self.boundary_i # internal point ind (K)
        intvals = []
        ijindex = []
        for i in internal_i:
            intvals.append(u[i])
            ijindex.append([i%self.nx, i//self.nx])
            
            
        logger.debug("Internal Values: %s", intvals)
        
        logger.debug("Internal index: %s",ijindex)
        
        logger.debug("Internal X Y values: %s", [[self.dx*val for val in sublist] for sublist in ijindex])
        self.solution = np.reshape(u, (self.ny, self.nx))
        
        
        
        
        

    def plot_solution(self):
        """
        Plot the PDE solution.
        """
        # TODO - your code here
        logger.info("Running Plot Solution")
        X, Y = np.meshgrid(self.x, self.y) # generate coordinates
        logger.debug("X meshgrid:\n%s", np.array2string(X, precision=3, suppress_small=True, max_line_width=120))
        logger.debug("Y meshgrid:\n%s", np.array2string(Y, precision=3, suppress_small=True, max_line_width=120))
        
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, self.solution, levels = 100, cmap = "viridis")
        fig.colorbar(c, ax = ax)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Mesh solution contour plot")
        plt.show()
        



class SolverHeatXT(object):
    """
    Class containing attributes and methods for solving the 1D heat equation. Assumes Dirichlet boundary conditions.

    Attributes:
        nx (int): number of mesh points along the x dimension.
        nt (int): number of mesh points along the t dimension.
        n (int): total number of mesh points.
        x (1D array): mesh coordinates along the x dimension.
        t (1D array): mesh coordinates along the t dimension.
        dx (float): mesh spacing along the x dimension.
        dt (float): mesh spacing along the t dimension.
        alpha (float): measure of thermal diffusivity in the heat equation.
        r (float): equal to alpha*dt/(dx^2), may be useful for diagnosing numerical stability.
        theta (float): weight applied to spatial derivative at t^(n+1), where 0 < theta <= 1.
        bc_x0 (dict): dictionary storing information for left boundary conditions.
        bc_x1 (dict): dictionary storing information for right boundary conditions.
        ic_t0 (dict): dictionary storing information for initial conditions.
        solution (2D array): solution array corresponding to mesh.

    Arguments:
        xlim (1D array): lower and upper limits in x dimension.
        tlim (1D array): lower and upper limits in t dimension.
        dx (float): desired mesh spacing in x dimension. May not exactly equal set mesh spacing.
        dt (float): desired mesh spacing in t dimension. May not exactly equal set mesh spacing.
        bc_x0 (dict): boundary conditions along x0.
        bc_x1 (dict): boundary conditions along x1.
        ic_t0 (dict): initial conditions at t0.
        alpha (float): measure of thermal diffusivity in the heat equation.
    """

    def __init__(self, xlim, tlim, dx, dt, alpha, theta, bc_x0, bc_x1, ic_t0):

        # TODO: define the integer number of mesh points, including boundaries, based on desired mesh spacing
        xmax = xlim[1]
        xmin = xlim[0]
        tmax = tlim[1]
        tmin = tlim[0]
        
        self.nx = int(round((xmax - xmin)/dx)) + 1
        self.nt = int(round((tmax - tmin)/dt)) + 1
        self.n = self.nx*self.nt
        
        # TODO: calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(xmin, xmax, self.nx)
        self.t = np.linspace(tmin, tmax, self.nt)
        
        # TODO: calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]


        # set ratio of step sizes, useful for examining numerical stability and implementing method
        self.alpha = alpha
        self.r = self.alpha*self.dt/(self.dx*self.dx)
        self.theta = theta

        # store the Dirichlet boundary conditions and initial conditions
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.ic_t0 = ic_t0

        # TODO: initialise solution matrix, apply the Dirichlet boundary and initial conditions to it now
        self.solution = np.zeros((self.nt, self.nx), dtype = float)
        
        if (self.ic_t0['type'] == 'initial'):
            self.solution[0,:] = self.ic_t0['function'](self.x, self.t[0])
            
        for ti in self.t:
            left_vals = np.array([self.bc_x0['function'](self.x[0], ti)], dtype = float)
            right_vals = np.array([self.bc_x1['function'](self.x[-1], ti)], dtype = float)
            
        self.solution[:, 0] = left_vals
        self.solution[:, -1] = right_vals
        
        self.internal_xi = np.arrange(1, self.nx-1)
        self.internal_ti = np.arrange(1, self.nt-1)

    def solve_explicit(self):
        """
        Solve the 1D heat equation using an explicit solution method.
        """
        # TODO - your code here
        r = self.r
        
        if not (0 < r <= 0.5):
            logger.critical("Explicit solution unstable unless 0<r<=0.5: %s", r)
        
        # iterate over time to get next steps in time
        for n in range(self.nt -1):
            u_n = self.solution[n, :]
            t_next = self.t[n+1]
            
            # dirichlet at next time
            self.solution[n+1, 0] = self.bc_x0['function'](self.x[0], t_next)
            self.solution[n+1, -1] = self.bc_x1['function'](self.x[-1], t_next)
            
            # update interior points
            self.solution[n+1, 1:-1] = r *u_n[0:-2] + (1-2*r) * u_n[1:-1] + self.r * u_n[2:]


       

    def implicit_update_a(self):
        """
        Set coefficients in the matrix A, prior to iterative solution. This only needs to be set once i.e. it doesn't
        change with each iteration, unlike the b vector.

        Returns:
            a (2D array): coefficient matrix for implicit method (dimension 2 nx by 2 nx)
        """
        # TODO - your code here
        nx = self.nx
        r = self.r
        theta = self.theta
        
        # build L array
        L = np.zeros((nx,nx), dtype = float)
        for i in range(nx):
            L[i,i] = -2.0
            if i -1>= 0:
                L[i,i-1] = 1.0
            if i+1 < nx:
                L[i, i+1] = 1.0
            
        A = np.zeros((2 *nx, 2 *nx), dtype = float)
        
        # top left
        for i in range(nx):
            A[i,i] = 1.0
        
        # bottom left
        A[nx:2*nx, 0:nx] = (1.0-theta) * r * L
        
        # bottom right
        A[nx:2*nx, nx:2*nx] = np.eye(nx, dtype = float) - theta * r * L
        
        # dirichlet at new time
        for i in (0, nx-1):
            row = nx + i
            A[row, 0:nx] = 0.0
            A[row, nx:2*nx] = 0.0
            A[row, nx + i] = 1.0
            
        self.A = A
        return A

    def implicit_update_b(self, i_t):
        """
        Update the b vector for the current time step to be solved, making use of the
        data already stored in self.solution.

        Arguments:
            i_t (int): time index for the current step being solved.

        Returns:
            b (1D array): vector of constants for implicit method (length of 2 nx)
        """
        # TODO - your code here
        nx = self.nx
        r = self.r
        theta = self.theta
        
        u_n = self.solution[n, :]
        t_np1 = self.t[n+1]
        x = self.x
        b = np.zeros(2*nx, dtype = float)
        
        b[0:nx] = u_n
        bottom = np.array(u_n, dtype = float)
        
        if nx >= 3:
            coeff = (1.0 - theta) * r
            # interior is
            for i in range(1, nx-1):
                lap_u = (u_n[i-1] - 2.0 * u_n[i] + u_n[i +1])
                bottom[i] = u_n[i] + coeff * lap_u

    def solve_implicit(self):
        """
        Solve the 1D heat equation using an implicit solution method.
        """
        # TODO - your code here
        pass

    def plot_solution(self, n_lines):
        """
        Plot the solution as a series of 1D line plots of u(x) at different t.

        Arguments:
            n_lines (int): number of time points to plot between t0 and t1 (inclusive)
        """
        # TODO - your code here
        pass


class SolverWaveXT(object):
    """
    Class containing attributes and methods useful for solving the 1D wave equation. Assumes Dirichlet boundary
    conditions.

    Attributes:
        nx (int): number of mesh points along the x dimension.
        nt (int): number of mesh points along the t dimension.
        n (int): total number of mesh points.
        x (1D array): mesh coordinates along the x dimension.
        t (1D array): mesh coordinates along the t dimension.
        dx (float): mesh spacing along the x dimension.
        dt (float): mesh spacing along the t dimension.
        c (float): c coefficient in the wave equation.
        r (float): ratio of time steps equal to (c*dt/dx)^2, useful for diagnosing stability.
        bc_x0 (dict): dictionary storing information regarding left boundary conditions.
        bc_x1 (dict): dictionary storing information regarding right boundary conditions.
        ic_t0 (dict): dictionary storing information regarding initial condition, u(x,t=0).
        ic_dt0 (dict): dictionary storing information regarding initial condition, du/dt(x,t=0).
        solution (2D array): solution array corresponding to mesh.

    Arguments:
        xlim (1D array): lower and upper limits in x dimension.
        tlim (1D array): lower and upper limits in t dimension.
        dx (float): desired mesh spacing in x dimension.
        dt (float): desired mesh spacing in t dimension.
        bc_x0 (dict): boundary conditions along x0.
        bc_x1 (dict): boundary conditions along x1.
        ic_t0 (dict): initial condition, u(x,t=0).
        ic_dt0 (dict): initial condition, du/dt(x,t=0).
        c (float): c coefficient in the wave equation.
    """

    def __init__(self, xlim, tlim, dx, dt, c, bc_x0, bc_x1, ic_t0, ic_dt0):

        # TODO: define the integer number of mesh points, including boundaries, based on desired mesh spacing

        # TODO: calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays

        # TODO: calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments

        # set ratio of step sizes, useful for examining numerical stability and implementing method
        self.c = c
        self.r = self.c * self.dt/self.dx

        # store the Dirichlet boundary conditions and initial conditions
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.ic_t0 = ic_t0
        self.ic_dt0 = ic_dt0

        # TODO: initialise solution matrix and apply boundary conditions

    def solve_explicit(self):
        """
        Solve the 1D wave equation using an explicit method.
        """
        # TODO - your code here
        pass

    def plot_solution(self, n_lines):
        """
        Plot the solution as a series of 1D line plots of u(x) at different t.

        Arguments:
            n_lines (int): number of time points to plot between t0 and t1 (inclusive)
        """
        # TODO - your code here
        pass


