from fenics import *
import matplotlib.pyplot as plt

# Création d'un maillage pour un domaine carré
mesh = UnitSquareMesh(8, 8)

# Définition de l'espace fonctionnel
V = FunctionSpace(mesh, 'P', 1)

# Conditions aux limites
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Définition de la fonction source
f = Constant(-6.0)

# Définition de la forme variationnelle
u = TrialFunction(V)
v = TestFunction(V)
a = (dot(grad(u), grad(v))+u*v)*dx
L = f*v*dx

# Résolution du problème
u_solution = Function(V)
solve(a == L, u_solution, bc)

# Affichage de la solution
c = plot(u_solution)
plt.colorbar(c)
plt.title("Solution de l'équation de Poisson")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
