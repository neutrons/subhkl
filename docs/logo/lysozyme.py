from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.spatial
from mantid.geometry import ReflectionGenerator
from mantid.kernel import V3D
from mantid.simpleapi import CreateSampleWorkspace, LoadCIF

r = 0.4
y_max = 0.2

a = 61.2
b = 61.2
c = 96.8
alpha = np.deg2rad(90)
beta = np.deg2rad(90)
gamma = np.deg2rad(120)

np.random.seed(13)

lamda_min, lamda_max = 2.8, 4.5
two_theta_min, two_theta_max = np.deg2rad(1), np.deg2rad(75)

G = np.array(
    [
        [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
        [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
        [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
    ]
)

G_inv = np.linalg.inv(G)

B = scipy.linalg.cholesky(G_inv)

U = scipy.spatial.transform.Rotation.random().as_matrix()

UB = np.dot(U, B)

h = np.arange(-100, 100 + 1, 1)
k = np.arange(-100, 100 + 1, 1)
l = np.arange(-100, 100 + 1, 1)

h, k, l = np.meshgrid(h, k, l)

h, k, l = h.flatten(), k.flatten(), l.flatten()

Qx, Qy, Qz = 2 * np.pi * np.einsum("ij,jk->ik", UB, [h, k, l])

Q = np.sqrt(Qx**2 + Qy**2 + Qz**2)

lamda = -4 * np.pi * Qz / Q**2

mask = (lamda > lamda_min) & (lamda < lamda_max)

Qx, Qy, Qz, Q, lamda = Qx[mask], Qy[mask], Qz[mask], Q[mask], lamda[mask]

h, k, l = h[mask], k[mask], l[mask]

d = 2 * np.pi / Q

two_theta = 2 * np.arcsin(0.5 * lamda / d)
az_phi = np.arctan2(Qy, Qx)

mask = (two_theta > two_theta_min) & (two_theta < two_theta_max)

Qx, Qy, Qz, Q, lamda = Qx[mask], Qy[mask], Qz[mask], Q[mask], lamda[mask]

h, k, l = h[mask], k[mask], l[mask]

d = 2 * np.pi / Q

two_theta = 2 * np.arcsin(0.5 * lamda / d)
az_phi = np.arctan2(Qy, Qx)

x_hat = np.sin(two_theta) * np.cos(az_phi)
y_hat = np.sin(two_theta) * np.sin(az_phi)
z_hat = np.cos(two_theta)

denom = np.sqrt(x_hat**2 + z_hat**2)

x = r * x_hat / denom
y = r * y_hat / denom
z = r * z_hat / denom

mask = (y > -y_max) & (y < y_max)

x, y, z = x[mask], y[mask], z[mask]

Qx, Qy, Qz, Q, lamda = Qx[mask], Qy[mask], Qz[mask], Q[mask], lamda[mask]

h, k, l = h[mask], k[mask], l[mask]

d = 2 * np.pi / Q

mask = d > 4

x, y, z = x[mask], y[mask], z[mask]

Qx, Qy, Qz, Q, lamda = Qx[mask], Qy[mask], Qz[mask], Q[mask], lamda[mask]

h, k, l = h[mask], k[mask], l[mask]

d = 2 * np.pi / Q

two_theta = 2 * np.arcsin(0.5 * lamda / d)

u = r * np.arctan2(x, z)
v = y.copy()

sample_ws = CreateSampleWorkspace()

LoadCIF(
    sample_ws,
    str(Path(__file__).with_name("5VNQ.cif")),
)

generator = ReflectionGenerator(sample_ws.sample().getCrystalStructure())

hkls = [V3D(*hkl) for hkl in np.column_stack([h, k, l]).tolist()]

fSquared = np.array(generator.getFsSquared(hkls))


fig, ax = plt.subplots(1, 1)
ax.scatter(
    u,
    v,
    s=np.log10(fSquared * 0.5 * lamda**4 / np.sin(0.5 * two_theta) ** 2),
    c=np.log10(fSquared * 0.5 * lamda**4 / np.sin(0.5 * two_theta) ** 2),
    cmap="binary",
    alpha=0.5,
    faceedgecolor="none",
)
ax.set_aspect(1)
ax.set_axis_off()
fig.savefig("5VNQ.pdf", bbox_inches="tight")
fig.show()
