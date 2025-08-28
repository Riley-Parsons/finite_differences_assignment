# import statements
from functions_fd import *
# NOTE that this code is set up for modelling the problem in Task 1B (which has assessed hand-ins).
# You are welcome to add a section for testing the problem in Task 1A if you like.

# define the Poisson function
def poisson(x, y):
    return x - y

# set up the x and y dimensions.
xlim = np.array([-2., 2.])
ylim = np.array([-3., 3.])

# set up the boundary conditions
bc_x0 = {'type': 'neumann', 'function': lambda x, y: x}
bc_x1 = {'type': 'neumann', 'function': lambda x, y: y}
bc_y0 = {'type': 'dirichlet', 'function': lambda x, y: x * y}
bc_y1 = {'type': 'dirichlet', 'function': lambda x, y: x * y - 1.}

# TODO: your code below to solve and plot the 2D Poisson equation
