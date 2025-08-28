# import statements
from functions_fd import *

# define the boundary and initial conditions
bc_x0 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
bc_x1 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
ic_t0 = {'type': 'initial', 'function': lambda x, t: np.piecewise(x, [x <= 0., x >= 5., 0. < x < 5.], [200., 200., 30.])}

# TODO - your code here to solve and plot the 1D heat equation
