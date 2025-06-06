import numpy as np
from fenics import *
import matplotlib.pyplot as plt

NP = 35

# Function to normalize data
def scaling(S, S_max, S_min):
    return (S - S_min) / (S_max - S_min)
# Function to inverse normalize data
def inverse_scaling(S, S_max, S_min):
    return (S_max - S_min) * S + S_min

#Load Data
Uh_POD=np.load("Uh_POD.npy")
Uh_FE=np.load("Uh_FE.npy")
Uh_NN=np.load("Uh_POD_NN.npy")
Urb_POD = np.dot(Brb, np.dot(Brb.T, S))  # Project the snapshots onto the reduced basis

# Step 3: Calculate the error
error = S - Urb_POD  # Error between high-fidelity and reduced-order solutions

# Step 4: Sum of squared errors
sum_squared_errors = np.sum(np.linalg.norm(error, axis=0) ** 2)

# Step 5: Compute the singular values of the snapshot matrix
_, sigma, _ = np.linalg.svd(S, full_matrices=False)

# Assume we keep the first N_rb singular values
N_rb = Brb.shape[1]
sum_discarded_eigenvalues = np.sum(sigma[N_rb:] ** 2)

print("Sum of Squared Errors:", sum_squared_errors)
print("Sum of Discarded Eigenvalues:", sum_discarded_eigenvalues)

# Step 6: Comparison
print("The sum of squared errors is consistent with the sum of discarded eigenvalues:", np.allclose(sum_squared_errors, sum_discarded_eigenvalues, atol=1e-5))
uh_POD_norm = scaling(Uh_POD, np.max(Uh_POD), np.min(Uh_POD))
uh_FE_norm = scaling(Uh_FE, np.max(Uh_FE), np.min(Uh_FE))
uh_NN_norm = scaling(Uh_NN, np.max(Uh_NN), np.min(Uh_NN))
# Compute the L2 error
l2_error_FE = np.linalg.norm(uh_NN_norm - uh_FE_norm)
l2_error_POD = np.linalg.norm(uh_NN_norm - uh_POD_norm)
print(f"L2 Error Finite Element: {l2_error_FE}")
print(f"L2 Error POD: {l2_error_POD}")


# Chargement du maillage et de l’espace V (selon ton projet)
mesh = UnitSquareMesh(NP, NP)  
V = FunctionSpace(mesh, "CG", 1)


# Transformation
Uh_POD_V = Function(V)
Uh_POD_V.vector().set_local(uh_POD_norm.flatten())

Uh_NN_V=Function(V)
Uh_NN_V.vector().set_local(uh_NN_norm.flatten())

Uh_FE_V=Function(V)
Uh_FE_V.vector().set_local(uh_FE_norm.flatten())

# Affichage avec backend matplotlib
p1 = plot(Uh_POD_V, title="POD solution", backend="matplotlib")

# Changer la colormap
p1.set_cmap("rainbow")
# Ajouter la colorbar et afficher
plt.colorbar(p1)
plt.show()
# Affichage avec backend matplotlib
p2 = plot(Uh_FE_V, title="FE solution", backend="matplotlib")

# Changer la colormap
p2.set_cmap("rainbow")
# Ajouter la colorbar et afficher
plt.colorbar(p2)
plt.show()
# Affichage avec backend matplotlib
p3 = plot(Uh_NN_V, title=" NN-POD solution", backend="matplotlib")

# Changer la colormap
p3.set_cmap("rainbow")
# Ajouter la colorbar et afficher
plt.colorbar(p3)
plt.show()

# Affichage avec backend matplotlib
p4 = plot(Uh_NN_V-Uh_FE_V, title=" Relative diff POD-NN and FE", backend="matplotlib")

# Changer la colormap
p4.set_cmap("rainbow")
# Ajouter la colorbar et afficher
plt.colorbar(p4)
plt.show()

