# import statements
from functions_fd import *

# define the boundary and initial conditions
x0 = {'type': 'dirichlet', 'function': lambda x, t: 0.}
x1 = {'type': 'dirichlet', 'function': lambda x, t: 0.5}
t0 = {'type': 'initial', 'function': lambda x, t: np.sin(3*np.pi*x) + x/2}
dt0 = {'type': 'initial_derivative', 'function': lambda x, t: 0.}

# TODO - your code here to solve and plot the 1D wave equation
xlim = np.array([0.0, 1.0])
tlim = np.array([0.0, 1.0])
dx = 0.01
dt = 0.005
c = 1.0
solver = SolverWaveXT(xlim, tlim, dx, dt, c, x0, x1, t0, dt0)
solver.solve_explicit()
solver.plot_solution(n_lines=6)