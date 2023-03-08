from fenics import *
from mshr import *

T = 5.0            # final time
num_steps = 250     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
R1 = Rectangle(Point(0., 0.), Point(5., 5.))
R2 = Rectangle(Point(7., 0.), Point(12., 5.))
R3 = Rectangle(Point(5., 1.8), Point(7., 3.2))
mesh = generate_mesh(R1 + R2 + R3, 32)

V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition

def boundary_1(x, on_boundary):
    return on_boundary and x[0] <= 7

bc = DirichletBC(V, Constant(0.85), boundary_1)

# Define initial value
u_0 = Expression('exp(-a*pow(x[0] - 2.5, 2) - a*pow(x[1] - 2.5, 2))',
                 degree=2, a=2)
u_n = interpolate(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0) 

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

vtkfile = File('heat_2/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    # Save solution to VTK
    vtkfile << (u, t)
    plot(u)

    # Update previous solution
    u_n.assign(u)
