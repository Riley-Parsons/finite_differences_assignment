import numpy as np
import matplotlib.pyplot as plt
import math


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
        
        # TODO: calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.y = np.linspace(ylim[0], ylim[1], self.ny)
        
        # TODO: calculate the actual mesh spacing in x and y (dx, dy), may differ slightly from delta if not exactly divisible
        # assumes nx and ny >1
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        
        # TODO: initialise linear algebra matrices (a,b)
        self.a = np.zeros((self.n,self.n))
        self.b = np.zeros(self.n)
        
        # store the four boundary conditions
        self.bc_x0 = bc_x0 #left
        self.bc_x1 = bc_x1 #right
        self.bc_y0 = bc_y0 #bottom
        self.bc_y1 = bc_y1 #top

        # equation corresponding to forcing function
        self.poisson_function = poisson_function

        # create solution matrix attribute
        self.solution = None

    def dirichlet(self):
        """
        Apply Dirichlet boundary conditions to update the corresponding elements of the A matrix and b vector for the
        mesh points along the Dirichlet boundaries.
        """
        # TODO - your code here
        
        def toint(lis):
            return [int(val) for val in lis]
                

        Bind = list(range(0, self.nx))
        Tind = list(range(self.n-self.nx, self.n))
        Lind = np.linspace(0, self.n-self.nx, self.ny)
        Lind = toint(Lind.tolist())
        Rind = np.linspace(self.nx-1, self.n-1, self.ny)
        Rind = toint(Rind.tolist())
            
        boundary_i = set(Bind+Tind+Lind+Rind)
        

        for k in boundary_i:
            self.a[k,:] = 0.0
            self.a[k,k] = 1.0
            if k in Bind and self.bc_y0['type'] == 'dirichlet':
                self.b[k] = self.bc_y0["function"](self.x[k%self.nx], self.y[0])
                
            elif k in Tind and self.bc_y1['type'] == 'dirichlet':
                self.b[k] = self.bc_y1['function'](self.x[k%self.nx], self.y[self.ny-1])
                
            elif k in Lind and self.bc_x0['type'] == 'dirichlet':
                self.b[k] = self.bc_x0['function'](self.x[0], self.y[k//self.nx])
                
            elif k in Rind and self.bc_x1['type'] == 'dirichlet':
                self.b[k] = self.bc_x1['function'](self.x[self.nx-1], self.y[k//self.nx])


    def neumann(self):
        """
        Apply Neumann boundary conditions to update the corresponding elements of the A matrix and b vector for the
        mesh points along the Neumann boundaries.
        """
        # TODO - your code here (Not needed for task 1A)
        pass

    def internal(self):
        """
        Apply FD stencil and Poisson equation to update the corresponding elements of the A matrix and b vector for the
        internal mesh points.
        """
        # TODO - your code here
        pass

    def solve(self):
        """
        Update A and b in the system of equations, Au=b, then solve the system for u and re-shape to store on
        the mesh as the numerical solution.
        """

        # TODO - your code here
        pass

    def plot_solution(self):
        """
        Plot the PDE solution.
        """
        # TODO - your code here
        pass


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

        # TODO: calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays

        # TODO: calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments

        # set ratio of step sizes, useful for examining numerical stability and implementing method
        self.alpha = alpha
        self.r = self.alpha*self.dt/(self.dx*self.dx)
        self.theta = theta

        # store the Dirichlet boundary conditions and initial conditions
        self.bc_x0 = bc_x0
        self.bc_x1 = bc_x1
        self.ic_t0 = ic_t0

        # TODO: initialise solution matrix, apply the Dirichlet boundary and initial conditions to it now


    def solve_explicit(self):
        """
        Solve the 1D heat equation using an explicit solution method.
        """
        # TODO - your code here
        pass

    def implicit_update_a(self):
        """
        Set coefficients in the matrix A, prior to iterative solution. This only needs to be set once i.e. it doesn't
        change with each iteration, unlike the b vector.

        Returns:
            a (2D array): coefficient matrix for implicit method (dimension 2 nx by 2 nx)
        """
        # TODO - your code here
        pass

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
        pass

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


