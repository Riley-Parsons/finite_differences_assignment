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

# 1A testing section
test_bx0 = {'type': 'dirichlet', 'function': lambda x, y: 0} #lhs
test_bx1 = {'type': 'dirichlet', 'function': lambda x, y: y*(1-y)} #rhs
test_by0 = {'type': 'dirichlet', 'function': lambda x, y: 0} #bottom
test_by1 = {'type': 'dirichlet', 'function': lambda x, y: 0} #top
spur_bx0 = {'type': 'neumann', 'function': lambda x, y: 108.3}

test_xlim = np.array([0, 1])
test_ylim = np.array([0,1])

def test_func(x,y):
    return 6*x*y*(1-y)-2*x**3

test_solver = SolverPoissonXY(test_xlim, test_ylim, 0.2, test_bx0, test_bx1, test_by0, test_by1, test_func)
test_solver.solve()
print(test_solver.solution)

#
test_solver.plot_solution()

# TODO: your code below to solve and plot the 2D Poisson equation

