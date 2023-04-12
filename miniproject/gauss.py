from dolfin import *
from mshr import *
from fenics import *
import numpy as np

c=20
omega = 400
mesh_1 = Rectangle(Point(-0.6, -0.3), Point(0.6, 0.3))
domain = mesh_1
mesh = generate_mesh(domain, 200)
P1 = FiniteElement("CG", "triangle", 1)
element =  MixedElement([P1, P1])
G = FunctionSpace(mesh, P1)
V = FunctionSpace(mesh, element)
dt = 0.00025; t = 0; T = 0.03
x_0 = 0.1
tol = 1e-14

# Определяем границы
class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -0.6, tol)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.6, tol)

class BoundaryY0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -0.3, tol)

class BoundaryY1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.3, tol)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(9999)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
n = FacetNormal(mesh)

# Задаем функции
u = TrialFunction(V)
v_re, v_im = TestFunction(V)

# Задаем граничные условия
f_re = Expression('0.1 * exp(-k * pow(x[1], 2) * x_0 / (2 * (pow(x[0], 2) + pow(x_0, 2)))) / sqrt(pow(x[0], 2) + pow(x_0, 2)) * cos(-k * pow(x[1], 2) * x[0] / (2 * (pow(x[0], 2) + pow(x_0, 2))) - atan(x_0 / x[0]))', k = omega / c, x_0 = x_0, degree=4)
f_im = Expression('0.1 * exp(-k * pow(x[1], 2) * x_0 / (2 * (pow(x[0], 2) + pow(x_0, 2)))) / sqrt(pow(x[0], 2) + pow(x_0, 2)) * sin(-k * pow(x[1], 2) * x[0] / (2 * (pow(x[0], 2) + pow(x_0, 2))) - atan(x_0 / x[0]))', k = omega / c, x_0 = x_0, degree=4)
bc = []
for i in range(4):
    bc.append(DirichletBC(V.sub(0), f_re, boundary_markers, i))
    bc.append(DirichletBC(V.sub(1), f_im, boundary_markers, i))

# Задаем формы
a = inner(grad(u[0]), grad(v_re)) * dx + 2 * (omega / c) * grad(u[1])[0] * v_re * dx  + grad(grad(u[0])[0])[0] * v_re * dx\
    + inner(grad(u[1]), grad(v_im)) * dx - 2 * (omega / c) * grad(u[0])[0] * v_im * dx  + grad(grad(u[1])[0])[0] * v_im * dx
L = Constant(0) * v_re * dx

# Решаем
u = Function(V)
solve(a == L, u, bc)

U1, U2 = u.split(deepcopy=True)
U_re = U1.vector().get_local()
U_im = U2.vector().get_local()
x_coords = ((np.array(mesh.coordinates()).T)[0]).T
nodes = np.array(vertex_to_dof_map(G))

# Делаем сдвиги по координате
for i, node in enumerate(nodes):
    U = (U_re[node]**2 + U_im[node]**2)**0.5
    arg = np.arctan2(U_im[node], U_re[node])
    arg -= omega / c * x_coords[i]
    U_re[node] = U * np.cos(arg)
    U_im[node] = U * np.sin(arg)

# Делаем сдвиги по времени и запись в файл
vtkfile = File('gauss/gauss.pvd')
arg = 0
while t <= T:
    t += dt
    arg += omega * dt
    W = U_re * np.cos(arg) - U_im * np.sin(arg)

    U1.vector().set_local(W)

    vtkfile << (U1, t)

    print("Current time: %f / %f" % (t, T))
