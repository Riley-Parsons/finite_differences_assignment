# import statements
from functions_fd import *

# define the boundary and initial conditions
bc_x0 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
bc_x1 = {'type': 'dirichlet', 'function': lambda x, t: 200.}
ic_t0 = {'type': 'initial', 'function': lambda x, t: np.piecewise(x, [x <= 0., x >= 5., 0. < x < 5.], [200., 200., 30.])}

# TODO - your code here to solve and plot the 1D heat equation
materials = {"silver": 1.5, "copper": 1.25, "aluminium":1}

dts = [0.001, 0.005]
theta = 0.5
xlim = np.array([0.0, 5.0])
tlim = np.array([0.0, 4.0])
dx = 0.1
alpha = materials['silver']

# silver rod
for dt in dts:
    # explicit
    solver = SolverHeatXT(xlim = xlim, tlim = tlim, dx = dx, dt = dt, alpha = alpha,theta=theta, bc_x0=bc_x0, bc_x1=bc_x1, ic_t0=ic_t0)
    solver.solve_explicit()
    solver.plot_solution(n_lines=4)
    
    # implicit crank nicolson, 0.5 theta
    solver = SolverHeatXT(xlim=xlim, tlim=tlim, dx=dx, dt=dt,alpha=alpha, theta=theta,bc_x0=bc_x0, bc_x1=bc_x1, ic_t0=ic_t0)
    solver.solve_implicit()
    solver.plot_solution(n_lines=4)
    
    
for name, alpha in materials.items():
    dt = 0.005  
    solver = SolverHeatXT(
        xlim=xlim, tlim=tlim, dx=dx, dt=dt,
        alpha=alpha, theta=theta,
        bc_x0=bc_x0, bc_x1=bc_x1, ic_t0=ic_t0
    )
    solver.solve_implicit()
    u_final= solver.solution[-1, :] # temperature along the rod at final time (t=4s)
    u_min = float(np.min(u_final))
    ok= (u_min >= 175.0)
    print(f"{name:8s} {u_min:12.3f} {ok}")