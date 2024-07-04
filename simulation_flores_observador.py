import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, cont2discrete, place_poles


# Define funções
def continuous_dynamics(xc, uc, Ac, Bc):
    return Ac @ xc + Bc @ uc

def discrete_dynamics(x, u, A, B):
    return A @ x + B @ u

def plot_tempo_continuo(vt, vx, vu, vtc, vxc, vuc):
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

def plot_tempo_discreto(vt, vx, vu):
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


##### Constantes e inicializações #####
CASE = 2

# Planta
Ac = np.array([[0, 1], [0, -2]])
Bc = np.array([[0], [2]])
Cc = np.array([[1, 0]])
sysc = lti(Ac, Bc, Cc, 0)
intstep = 1e-3
hc = intstep

##### Discretização #####
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

##### Controlador #####
Pc = np.array([-2, -3])
Kc = place_poles(Ac, Bc, Pc).gain_matrix

P = np.array([-0.1, 0.5])
K = place_poles(A, B, P).gain_matrix

##### Observador #####
P_observer = np.array([-0.001, 0.005])
Pc_observer = P_observer

# Projeto do observador de estados para o sistema contínuo
Lc = place_poles(Ac.T, Cc.T, Pc_observer).gain_matrix.T

# Projeto do observador de estados para o sistema discreto
L = place_poles(A.T, C.T, P_observer).gain_matrix.T

##### Logs #####
## Contínuo
vtc, vxc, vuc = [], [], []
vthc, vxhc, vuhc = [], [], []

# Contínuo com Observador
vtc_hat, vxc_hat, vuc_hat = [], [], []
vthc_hat, vxhc_hat, vuhc_hat = [], [], []

## Discreto
vt, vx, vu = [], [], []

# Discreto com Observador
vt_hat, vx_hat, vu_hat = [], [], []

##### Simulação #####
# Contínuo
xc = np.array([0.5, 0.5]).reshape(-1, 1)
xhc = xc
xc_hat = np.array([0.8, 0.8]).reshape(-1, 1) # Inicialização do estado estimado
xhc_hat = xc_hat

# Discreto
x = xc
x_hat = xhc_hat  
tc = 0
t = tc

while t < 10:
    # u = np.sin(2 * np.pi * t / 50)

    uhc = -K @ xhc
    uhc_hat = -K @ xhc_hat 

    u = -K @ x  
    u_hat = -K @ x_hat 

    ## Logs
    # Log pontos discretos do contínuo
    vthc.append(tc)
    vuhc.append(uhc.item())  # Convertendo para um valor escalar
    vxhc.append(xhc.flatten())  # Convertendo para um vetor 1D

    # Log pontos discretos do discreto
    vt.append(t)
    vu.append(u.item())  
    vx.append(x.flatten()) 

    # Log pontos discretos do contínuo com observador
    vuhc_hat.append(uhc_hat.item())  
    vxhc_hat.append(xhc_hat.flatten())  

    # Log pontos discretos do discreto com observador
    vu_hat.append(u_hat.item())  
    vx_hat.append(x_hat.flatten()) 

    #### Próximo estado ####

    ### 1 - MODO CONTÍNUO ###
    uc = uhc  # ZOH
    uc_hat = uhc_hat
    for _ in range(substeps):
        # Log pontos contínuos
        vtc.append(tc)
        vuc.append(uc.item())  # Convertendo para um valor escalar
        vxc.append(xc.flatten())  # Convertendo para um vetor 1D

        # Com observador
        vuc_hat.append(uc_hat.item()) 
        vxc_hat.append(xc_hat.flatten())

        # Integração
        xc = xc + hc * continuous_dynamics(xc, uc, Ac, Bc)
        # Atualizando o observador
        y = Cc @ xc
        xc_hat = xc_hat + hc * continuous_dynamics(xc_hat, uc_hat, Ac, Bc) + hc * Lc @ (y - Cc @ xc_hat)
        tc += hc

    xhc = xc
    xhc_hat = xc_hat
 
    ### 2 - MODO DISCRETO ### 
        
    # Atualizando o estado real
    x = discrete_dynamics(x, u, A, B)
    # Atualizando o observador
    y = C @ x
    x_hat = A @ x_hat + B @ u + L @ (y - C @ x_hat)
    t += h

# Convertendo listas para arrays numpy para plotagem
vx = np.array(vx)
vxhc = np.array(vxhc)
vxc = np.array(vxc)
vx_hat = np.array(vx_hat)
vxhc_hat = np.array(vxhc_hat)
vxc_hat = np.array(vxc_hat)

# Plotagem 

# 1 - Tempo contínuo.
plt.figure(1)
plt.title("Tempo contínuo")
plot_tempo_continuo(vthc, vxhc, vuhc, vtc, vxc, vuc)

plt.figure(2)
plt.title(f"Tempo contínuo com Observador. Pólos observador = {P_observer}")
plot_tempo_continuo(vthc, vxhc_hat, vuhc_hat, vtc, vxc_hat, vuc_hat)


# 2 - Tempo discreto.
plt.figure(3)
plt.title(f"Tempo Discreto")
plot_tempo_discreto(vt, vx, vu)

plt.figure(4)
plt.title(f"Tempo Discreto com Observador. Pólos observador = {Pc_observer}")
plot_tempo_discreto(vt, vx_hat, vu_hat)

plt.show()
