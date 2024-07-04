import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, cont2discrete, place_poles

# Define funções
def continuous_dynamics(xc, uc, Ac, Bc):
    return Ac @ xc + Bc @ uc

def discrete_dynamics(x, u, A, B):
    return A @ x + B @ u

# Constantes e inicializações
CASE = 2

# Planta
Ac = np.array([[0, 1], [0, -2]])
Bc = np.array([[0], [2]])
Cc = np.array([[1, 0]])
sysc = lti(Ac, Bc, Cc, 0)
intstep = 1e-3
hc = intstep

# Discretização
h = 0.1
substeps = int(h / hc)
TypeDiscretization = 'ZOH'

if TypeDiscretization == 'Euler':
    # Discretização Euler
    A = np.eye(2) + h * Ac
    B = h * Bc
    C = Cc
elif TypeDiscretization == 'ZOH':
    # Discretização ZOH
    sysZOH = cont2discrete((Ac, Bc, Cc, 0), h, method='zoh')
    A, B, C, D, _ = sysZOH

# Controlador
Pc = np.array([-2, -3])
Kc = place_poles(Ac, Bc, Pc).gain_matrix

P = np.array([-0.1, 0.5])
K = place_poles(A, B, P).gain_matrix

# Logs
vtc, vxc, vuc = [], [], []
vt, vx, vu = [], [], []

# Simulação
xc = np.array([0.5, 0.5]).reshape(-1, 1)
x = xc
tc = 0
t = tc

while t < 10:
    # u = np.sin(2 * np.pi * t / 50)
    u = -K @ x
    # Log de dados
    vt.append(t)
    vu.append(u.item())  # Convertendo para um valor escalar
    vx.append(x.flatten())  # Convertendo para um vetor 1D

    # Próximo estado
    if CASE == 1:
        uc = u  # ZOH
        for _ in range(substeps):
            # Log de dados
            vtc.append(tc)
            vuc.append(uc.item())  # Convertendo para um valor escalar
            vxc.append(xc.flatten())  # Convertendo para um vetor 1D

            # Integração
            xc = xc + hc * continuous_dynamics(xc, uc, Ac, Bc)
            tc += hc
        x = xc
        t = tc
    elif CASE == 2:
        x = discrete_dynamics(x, u, A, B)
        t += h

# Convertendo listas para arrays numpy para plotagem
vx = np.array(vx)
vxc = np.array(vxc)

# Plotagem
plt.figure(1)

if CASE == 1:
    plt.subplot(3, 1, 1)
    plt.plot(vt, vx[:, 0], '*k', linewidth=1)
    if vtc:
        plt.plot(vtc, vxc[:, 0], 'g', linewidth=3)
    plt.grid(True)
    plt.ylabel('x1')

    plt.subplot(3, 1, 2)
    plt.plot(vt, vx[:, 1], '*k', linewidth=1)
    if vtc:
        plt.plot(vtc, vxc[:, 1], 'g', linewidth=3)
    plt.grid(True)
    plt.ylabel('x2')

    plt.subplot(3, 1, 3)
    plt.plot(vt, vu, '*k', linewidth=1)
    if vtc:
        plt.plot(vtc, vuc, 'g', linewidth=3)
    plt.grid(True)
    plt.ylabel('u')
elif CASE == 2:
    plt.subplot(3, 1, 1)
    plt.plot(vt, vx[:, 0], 'om', linewidth=1)
    plt.grid(True)
    plt.ylabel('x1')

    plt.subplot(3, 1, 2)
    plt.plot(vt, vx[:, 1], 'om', linewidth=1)
    plt.grid(True)
    plt.ylabel('x2')

    plt.subplot(3, 1, 3)
    plt.plot(vt, vu, 'om', linewidth=1)
    plt.grid(True)
    plt.ylabel('u')

plt.show()
