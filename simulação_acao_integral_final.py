import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, cont2discrete, place_poles
import sys


def continuous_dynamics(xc, uc, Ac, Bc):
    return Ac @ xc + Bc @ uc

def discrete_dynamics(x, u, A, B):
    return A @ x + B @ u



def plot_i_continuo(fig, vt, vu, vtc, vuc, title):
    pass

def plot_ii_continuo(fig, vt, vu, vu_hat, vtc, vuc, vuc_hat, title):

    axs = fig.subplots(2, 1, sharex=True)

    fig.suptitle(title)

    axs[0].plot(vt, vu, '*k', linewidth=1)
    axs[0].plot(vtc, vuc, 'g', linewidth=3)
    axs[0].grid(True)
    axs[0].set_ylabel('v')

    axs[1].plot(vt, vu_hat, '*k', linewidth=1)
    axs[1].plot(vtc, vuc_hat, 'g', linewidth=3)
    axs[1].grid(True)
    axs[1].set_ylabel('v_hat')

    plt.tight_layout()


def plot_iii_continuo(fig, vt, vy, vy_hat, vtc, vyc, vyc_hat, title):
    
    axs = fig.subplots(2, 1, sharex=True)
    
    fig.suptitle(title)
    
    axs[0].plot(vt, vy, '*k', linewidth=1)
    axs[0].plot(vtc, vyc, 'g', linewidth=3)
    axs[0].grid(True)
    axs[0].set_ylabel('y')

    axs[1].plot(vt, vy_hat, '*k', linewidth=1)
    axs[1].plot(vtc, vyc_hat, 'g', linewidth=3)
    axs[1].grid(True)
    axs[1].set_ylabel('y_hat')

    plt.tight_layout()


def plot_iv_continuo(fig, vt, vx, vx_hat, vtc, vxc, vxc_hat, title):
    
    axs = fig.subplots(4, 1, sharex=True)
    
    fig.suptitle(title)

    axs[0].plot(vt, vx[:, 0], '*k', linewidth=1, label='Discreto')
    axs[0].plot(vtc, vxc[:, 0], 'g', linewidth=3, label='Contínuo')
    axs[0].grid(True)
    axs[0].set_ylabel('x1')
    axs[0].legend()

    axs[1].plot(vt, vx_hat[:, 0], '*k', linewidth=1)
    axs[1].plot(vtc, vxc_hat[:, 0], 'g', linewidth=3)
    axs[1].grid(True)
    axs[1].set_ylabel('x1_hat')

    axs[2].plot(vt, vx[:, 1], '*k', linewidth=1)
    axs[2].plot(vtc, vxc[:, 1], 'g', linewidth=3)
    axs[2].grid(True)
    axs[2].set_ylabel('x2')

    axs[3].plot(vt, vx_hat[:, 1], '*k', linewidth=1)
    axs[3].plot(vtc, vxc_hat[:, 1], 'g', linewidth=3)
    axs[3].grid(True)
    axs[3].set_ylabel('x2_hat')

    plt.tight_layout()


def plot_v_continuo(fig, vt, vxi, vtc, vxic, title):
    
    axs = fig.subplots(1, 1, sharex=True)

    fig.suptitle(title)

    axs.plot(vt, vxi, '*k', linewidth=1)
    axs.plot(vtc, vxic, 'g', linewidth=3)
    axs.grid(True)
    axs.set_ylabel('xi')

    plt.tight_layout()


def plot_tempo_discreto(fig, vt, vx, vxi, vu, vy, title):
    
    axs = fig.subplots(5, 1, sharex=True)
    
    fig.suptitle(title)
    
    axs[0].plot(vt, vx[:, 0], 'om', linewidth=1)
    axs[0].grid(True)
    axs[0].set_ylabel('x1')

    axs[1].plot(vt, vx[:, 1], 'om', linewidth=1)
    axs[1].grid(True)
    axs[1].set_ylabel('x2')

    axs[2].plot(vt, vxi, 'om', linewidth=1)
    axs[2].grid(True)
    axs[2].set_ylabel('xi')

    axs[3].plot(vt, vy, 'om', linewidth=1)
    axs[3].grid(True)
    axs[3].set_ylabel('y')

    axs[4].plot(vt, vu, 'om', linewidth=1)
    axs[4].grid(True)
    axs[4].set_ylabel('u')


a = 1
b = 1.5
mi = 1

'''
item (c)

# Parâmetros do sistema
a = np.random.uniform(-2, 5)
b = np.random.uniform(1, 2)
c = np.random.uniform(3, 5)
d = -3  # Perturbação constante
mi = np.random.uniform(0.1, 1.5)
'''


# Planta
Ac = np.array([[0, 1], [a/mi, (a*mi - 1)/mi]])
Bc = np.array([[0], [-a*b/mi]])
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


print("Ac =\n", Ac)
print("A =\n", A)
print("\nBc =\n", Bc)
print("B =\n", B)

print("eig(A) =", np.linalg.eig(A))
print("eig(Ac) =", np.linalg.eig(Ac))

##### Controlador Ação Integral #####

def matrizes_aumentadas(A, B, C):

    # Cria uma matriz de zeros com as dimensões apropriadas
    zeros_2x1 = np.zeros((len(A), 1))
    zeros_1x1 = np.zeros((1, 1))

    # Construir a matriz aumentada Aa
    Aa = np.block([[A, zeros_2x1],
                   [-C, zeros_1x1]])
    
    Ba = np.block([[B], 
                   [zeros_1x1]])

    Bm = np.block([[zeros_2x1],
                   [1]])

    Ca = np.block([[C, zeros_1x1]])

    return Aa, Ba, Bm, Ca


ym = -10

# Exibir a matriz aumentada Aa
Aa, Ba, Bm, Ca = matrizes_aumentadas(A, B, C)

# Exibir a matriz aumentada Aac
Aac, Bac, Bmc, Cac = matrizes_aumentadas(Ac, Bc, Cc)


tau = 1.5

Pc = np.array([-tau, -0.411871*tau, -20])
Kac = place_poles(Aac, Bac, Pc).gain_matrix 
Kc = np.expand_dims(Kac[0][:len(A)], axis=0)
kic = -Kac[0][-1]


print("Aac - Bac*Kac =\n", Aac - Bac @ Kac)
print("eig(Aac - Bac*Kac) =", np.linalg.eig(Aac - Bac*Kac))


P = np.array([-0.1, 0.5, 0.55])
Ka = place_poles(Aa, Ba, P).gain_matrix
K = np.expand_dims(Ka[0][:len(A)], axis=0)
ki = -Ka[0][-1]


##### Observador #####
P_observer = np.array([-0.001, 0.005])
Pc_observer = P_observer

# Projeto do observador de estados para o sistema contínuo
Lc = place_poles(Ac.T, Cc.T, Pc_observer).gain_matrix.T

# Projeto do observador de estados para o sistema discreto
L = place_poles(A.T, C.T, P_observer).gain_matrix.T

##### Logs #####
## Contínuo
vtc, vxc, vuc, vxic, vyc = [], [], [], [], []
vthc, vxhc, vuhc, vxihc, vyhc = [], [], [], [], []

# Contínuo com Observador
vtc_hat, vxc_hat, vuc_hat, vyc_hat = [], [], [], []
vthc_hat, vxhc_hat, vuhc_hat, vyhc_hat = [], [], [], []

## Discreto
vt, vx, vxi, vu, vy = [], [], [], [], []

# Discreto com Observador
vt_hat, vx_hat, vu_hat, vy_hat = [], [], [], []

##### Simulação #####
# Contínuo
xc = np.array([0.5, 0.5]).reshape(-1, 1)
xic = 0.5       # aleatório
yc = 0.3        # aleatório
yc_hat = 0.8    # aleatório
yhc = yc
yhc_hat = yc_hat
xhc = xc
xihc = xic

xc_hat = np.array([0.8, 0.8]).reshape(-1, 1) # Inicialização do estado estimado
xhc_hat = xc_hat

# Discreto
x = xc
xi = xic
y = yc 
y_hat = yc_hat
x_hat = xhc_hat  
tc = 0
t = tc


while t < 10:
    uhc = -K @ xhc + kic*xic
    uhc_hat = -K @ xhc_hat + kic*xic

    u = -K @ x + ki*xi
    u_hat = -K @ x_hat + ki*xi
    
    ## Logs
    # Log pontos discretos do contínuo
    vthc.append(tc)
    vuhc.append(uhc.item())  # Convertendo para um valor escalar
    vxhc.append(xhc.flatten())  # Convertendo para um vetor 1D
    vxihc.append(xihc) 
    vyhc.append(yhc)

    # Log pontos discretos do discreto
    vt.append(t)
    vu.append(u.item())  
    vx.append(x.flatten()) 
    vxi.append(xi) 
    vy.append(y)

    # Log pontos discretos do contínuo com observador
    vuhc_hat.append(uhc_hat.item())  
    vxhc_hat.append(xhc_hat.flatten())  
    vyhc_hat.append(yhc_hat) 

    # Log pontos discretos do discreto com observador
    vu_hat.append(u_hat.item())  
    vx_hat.append(x_hat.flatten()) 
    vy_hat.append(y_hat) 

    #### Próximo estado ####

    ### 1 - MODO CONTÍNUO ###
    uc = uhc  # ZOH
    uc_hat = uhc_hat
    for _ in range(substeps):
        # Log pontos contínuos
        vtc.append(tc)
        vuc.append(uc.item())  # Convertendo para um valor escalar
        vxc.append(xc.flatten())  # Convertendo para um vetor 1D
        vxic.append(xic)  
        vyc.append(yc)  

        # Com observador
        vuc_hat.append(uc_hat.item()) 
        vxc_hat.append(xc_hat.flatten())
        vyc_hat.append(yc_hat)  

        # Integração
        xc = xc + hc * continuous_dynamics(xc, uc, Ac, Bc)
        
        # Atualizando o observador
        yc = (Cc @ xc)[0][0]
        xc_hat = xc_hat + hc * continuous_dynamics(xc_hat, uc_hat, Ac, Bc) + hc * Lc @ (yc - Cc @ xc_hat)
        yc_hat = (Cc @ xc_hat)[0][0]

        xic = xic + hc*(ym - yc)
        
        tc += hc

    xhc = xc
    xihc = xic
    xhc_hat = xc_hat
    yhc_hat = yc_hat
    yhc = yc
 
    ### 2 - MODO DISCRETO ### 
        
    # Atualizando o estado real
    x = discrete_dynamics(x, u, A, B)
    y = (C @ x)[0][0]
    xi = xi + h*(ym - y)
    
    # Atualizando o observador
    x_hat = A @ x_hat + B @ u + L @ (y - C @ x_hat)
    y_hat = (C @ x_hat)[0][0]
    t += h

# Convertendo listas para arrays numpy para plotagem
vx = np.array(vx)
vxhc = np.array(vxhc)
vxc = np.array(vxc)
vx_hat = np.array(vx_hat)
vxhc_hat = np.array(vxhc_hat)
vxc_hat = np.array(vxc_hat)
vxi = np.array(vxi)

# Plotagem 

# Plot (ii)
fig2 = plt.figure(2, figsize=(5, 6))
title = "Entrada v"
plot_ii_continuo(fig2, vthc, vuhc, vuhc_hat, vtc, vuc, vuc_hat, title)


# Plot (iii)

fig3 = plt.figure(3, figsize=(5, 6))
title = "Saída y"
plot_iii_continuo(fig3, vthc, vyhc, vyhc_hat, vtc, vyc, vyc_hat, title)

# Plot (iv)
fig4 = plt.figure(4, figsize=(5, 6))
title = "Estados da planta e do observador"
plot_iv_continuo(fig4, vthc, vxhc, vxhc_hat, vtc, vxc, vxc_hat, title)

#Plot (v)

fig5 = plt.figure(5, figsize=(5, 6))
title = "Estados do integrador"
plot_v_continuo(fig5, vthc, vxihc, vtc, vxic, title)

# 1 - Tempo contínuo.

'''''''''
fig1 = plt.figure(1, figsize=(5, 6))
title = "Tempo contínuo com Ação integral."
plot_tempo_continuo(fig1, vthc, vxhc, vxihc, vuhc, vyhc, vtc, vxc, vxic, vuc, vyc, title)

fig2 = plt.figure(2, figsize=(5, 6))
title = f"Tempo contínuo com Ação Integral e Observador. \nPólos observador = {P_observer}"
plot_tempo_continuo(fig2, vthc, vxhc_hat, vxihc, vuhc_hat, vyhc_hat, vtc, vxc_hat, vxic, vuc_hat, vyc_hat, title)

# 2 - Tempo discreto.
fig3 = plt.figure(3, figsize=(5, 6))
title = f"Tempo Discreto com Ação integral"
plot_tempo_discreto(fig3, vt, vx, vxi, vu, vy, title)

fig4 = plt.figure(4, figsize=(5, 6))
title = f"Tempo Discreto com Ação Integral e Observador. \nPólos observador = {Pc_observer}"
plot_tempo_discreto(fig4, vt, vx_hat, vxi, vu_hat, vy_hat, title)
'''
plt.show()
