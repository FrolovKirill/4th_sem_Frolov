from dolfin import *
from mshr import *
from fenics import *
import numpy as np

c=20
omega = 600
mesh_1 = Circle(Point(0.,0.), 0.6)
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
class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]**2 + x[1]**2 > 0.3

class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]**2 + x[1]**2 < 0.004

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(9999)
bx_outer = BoundaryOuter()
bc_inner = BoundaryInner()
bx_outer.mark(boundary_markers, 0)
bc_inner.mark(boundary_markers, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Задаем функции
u = TrialFunction(V)
v_re, v_im = TestFunction(V)

# Задаем граничные условия
bc_re = DirichletBC(V.sub(0), cos(omega * t), boundary_markers, 1)
bc_im = DirichletBC(V.sub(1), sin(omega * t), boundary_markers, 1)
bc = [bc_re, bc_im]

p_re = Expression("0.05 * cos(-om / c * (sqrt(pow(x[0], 2) + pow(x[1], 2)) - 0.05) + om * t) / sqrt(pow(x[0], 2) + pow(x[1], 2))", om = omega, t = t, c = c, degree=2, tol=tol)
p_im = Expression("0.05 * sin(-om / c * (sqrt(pow(x[0], 2) + pow(x[1], 2)) - 0.05) + om * t) / sqrt(pow(x[0], 2) + pow(x[1], 2))", om = omega, t = t, c = c, degree=2, tol=tol)
g_re = interpolate(p_re, G)
g_im = interpolate(p_im, G)

integrals_R_L = (grad(g_re)[0]**2 + grad(g_re)[1]**2)**0.5 * v_re * ds(0) - (omega / c) * g_im * v_re * ds(0)
integrals_R_L += (grad(g_im)[0]**2 + grad(g_im)[1]**2)**0.5 * v_im * ds(0) + (omega / c) * g_re * v_im * ds(0)
integrals_R_a = -(omega / c) * u[1] * v_re * ds(0)
integrals_R_a += (omega / c) * u[0] * v_im * ds(0)

# Задаем формы
a = inner(grad(u[0]), grad(v_re)) * dx - (omega / c)**2 * u[0] * v_re * dx \
    + inner(grad(u[1]), grad(v_im)) * dx - (omega / c)**2 * u[1] * v_im * dx\
    + integrals_R_a
L = integrals_R_L

# Решаем
u = Function(V)
solve(a == L, u, bc)

U1, U2 = u.split(deepcopy=True)
U_re = U1.vector().get_local()
U_im = U2.vector().get_local()

# Делаем сдвиги по времени и запись в файл
vtkfile = File('circ/helm_circ.pvd')
arg = 0
while t <= T:
    t += dt
    arg += omega * dt
    W = U_re * np.cos(arg) - U_im * np.sin(arg)

    U1.vector().set_local(W)

    vtkfile << (U1, t)

    print("Current time: %f / %f" % (t, T))