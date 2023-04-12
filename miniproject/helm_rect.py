from dolfin import *
from mshr import *
from fenics import *
import numpy as np

c=20
omega = 400
mesh_1 = Rectangle(Point(-0.6, -0.6), Point(0.6, 0.6))
mesh_2 = Circle(Point(0.,0.), 0.05)
domain = mesh_1 - mesh_2
mesh = generate_mesh(domain, 240)
P1 = FiniteElement("CG", "triangle", 1)
element =  MixedElement([P1, P1])
G = FunctionSpace(mesh, P1)
V = FunctionSpace(mesh, element)
dt = 0.00025; t = 0; T = 0.03
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
        return on_boundary and near(x[1], -0.6, tol)

class BoundaryY1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.6, tol)

class BoundaryCirc(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]**2 + x[1]**2 < 0.004

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(9999)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()
bcr = BoundaryCirc()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)
bcr.mark(boundary_markers, 4)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
n = FacetNormal(mesh)

# Задаем функции
u = TrialFunction(V)
v_re, v_im = TestFunction(V)

# Задаем граничные условия
bc_re = DirichletBC(V.sub(0), cos(omega * t - np.pi / 2), boundary_markers, 4)
bc_im = DirichletBC(V.sub(1), sin(omega * t - np.pi / 2), boundary_markers, 4)
bc = [bc_re, bc_im]

p_re = Expression("0.1 * cos(-k * sqrt(pow(x[0], 2) + pow(x[1], 2)) + 0.05 * k + om * t - pi / 2) / sqrt(pow(x[0], 2) + pow(x[1], 2))", k = omega / c, om = omega, t = t, pi = np.pi, degree=2)
p_im = Expression("0.1 * sin(-k * sqrt(pow(x[0], 2) + pow(x[1], 2)) + 0.05 * k + om * t - pi / 2) / sqrt(pow(x[0], 2) + pow(x[1], 2))", k = omega / c, om = omega, t = t, pi = np.pi, degree=2)
g_re = interpolate(p_re, G)
g_im = interpolate(p_im, G)

integrals_R_L = []
integrals_R_a = []
for i in range(4):
    integrals_R_L.append(dot(grad(g_re), n) * v_re * ds(i) - (omega / c) * g_im * v_re * ds(i))
    integrals_R_L.append(dot(grad(g_im), n) * v_im * ds(i) + (omega / c) * g_re * v_im * ds(i))
    integrals_R_a.append(-(omega / c) * u[1] * v_re * ds(i))
    integrals_R_a.append((omega / c) * u[0] * v_im * ds(i))

# Задаем формы
a = inner(grad(u[0]), grad(v_re)) * dx - (omega / c)**2 * u[0] * v_re * dx \
    + inner(grad(u[1]), grad(v_im)) * dx - (omega / c)**2 * u[1] * v_im * dx\
    + sum(integrals_R_a)
L = sum(integrals_R_L)

# Решаем
u = Function(V)
solve(a == L, u, bc)

U1, U2 = u.split(deepcopy=True)
U_re = U1.vector().get_local()
U_im = U2.vector().get_local()

# Делаем сдвиги по времени и запись в файл
vtkfile = File('rect/helm_cube.pvd')
arg = 0
while t <= T:
    t += dt
    arg += omega * dt
    W = U_re * np.cos(arg) - U_im * np.sin(arg)

    U1.vector().set_local(W)

    vtkfile << (U1, t)

    print("Current time: %f / %f" % (t, T))
