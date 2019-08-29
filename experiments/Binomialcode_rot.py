# %pylab inline
import os
import sys
import inspect
import numpy as np
from scipy.special import factorial
import h5py
from qoc.helper_functions.grape_functions import *
from qoc.main_grape.grape import Grape

data_path = '../pulses/output_pulses/'
initial_pulse = '../pulses/example_pulses/transmon_cat_initial_pulse.h5'


# Choose optimizing State transfer or Unitary gate
state_transfer = True

#Defining time scales
total_time = 1000.0
steps = 500
#Defining H0
qubit_state_num = 3  # changed from 4 to 2 to test code faster

alpha = 0.139
freq_ge = 0  # GHz, in rotating frame
ens = np.array([2*np.pi*ii*(freq_ge - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
Q_x = np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1)+np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1)
Q_y = (0+1j) * (np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1)-np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1))
Q_z = np.diag(np.arange(0, qubit_state_num))
I_q = np.identity(qubit_state_num)
H_q = np.diag(ens)

mode_state_num = 4  # changed from 5 to 3 to test code faster
chi = -0.565e-3
kappa = -chi / 10.0
mode_freq = 0  # GHz, in rotating frame
mode_ens = np.array([2*np.pi*ii*(mode_freq - 0.5*(ii-1)*kappa) for ii in np.arange(mode_state_num)])
M_x = np.diag(np.sqrt(np.arange(1, mode_state_num)), 1)+np.diag(np.sqrt(np.arange(1, mode_state_num)), -1)
M_y = (0+1j) * (np.diag(np.sqrt(np.arange(1, mode_state_num)), 1)-np.diag(np.sqrt(np.arange(1, mode_state_num)), -1))
M_z = np.diag(np.arange(0, mode_state_num))
H_m = np.diag(mode_ens)
I_m = np.identity(mode_state_num)

H0 = np.kron(H_q, I_m) + np.kron(I_q, H_m) + 2 * 2 * np.pi * chi * np.kron(Q_z, M_z)

#Defining control Hs

XI = np.kron(Q_x, I_m)
YI = np.kron(Q_y, I_m)
ZI = np.kron(Q_z, I_m)
IX = np.kron(I_q, M_x)
IY = np.kron(I_q, M_y)
IZ = np.kron(I_q, M_z)

Hops = [XI, YI, IX, IY]
ops_max_amp = [0.05*2*np.pi, 0.05*2*np.pi, 0.01*2*np.pi, 0.01*2*np.pi]
Hnames =['xi', 'yi', 'ix', 'iy']

#Defining the coherent state

def coherent_state(beta):
    coeffs = []
    pre_factor = np.exp((-(np.abs(beta))**2) / 2)
    for ii in range(mode_state_num):
        coeff = pre_factor * beta**ii / (np.sqrt(factorial(ii)))
        coeffs.append(coeff)
    return coeffs

def binomial_code_state(s):
    coeffs = np.zeros(mode_state_num)
    if s == 1:
        coeffs[0] = 1/np.sqrt(2)
        coeffs[4] = 1/np.sqrt(2)
    else:
        coeffs[2] = 1 # temporarily modified to test code faster, for binomial code should be coeffs[2]
    return coeffs

cat = binomial_code_state(-1)

print(cat)
cat = np.append(cat, np.zeros((qubit_state_num-1) * mode_state_num))

a = 0
for k in cat:
    a = a + np.abs(k)**2

#Defining dressed info

is_dressed = False

w_c, v_c, dressed_id = get_dressed_info(H0)

#dressed_info = {'dressed_id':dressed_id, 'eigenvectors':v_c,\
#               'eigenvalues':w_c, 'is_dressed':is_dressed}

dressed_info = None

#Defining states to include in the drawing of occupation
# states_draw_list = range(mode_state_num)
# states_draw_names = []
# for ii in range(mode_state_num):
#     states_draw_names.append('0_'+str(ii))


#Defining target

from quantum_optimal_control.helper_functions.grape_functions import get_state_index
g0 = v_c[:, get_state_index(0, dressed_id)]

if is_dressed:
    cat_d = np.zeros(mode_state_num*qubit_state_num)

    for ii in enumerate(cat):
        cat_d = cat_d + cat[ii] * v_c[:, get_state_index(ii, dressed_id)]
else:
    cat_d = cat
print("cat_d inner product: " + str(np.inner(cat_d, cat_d)))
print("g0 inner product: " + str(np.inner(g0, g0)))
U = [cat_d]


#Defining Concerned states (starting states)
g = np.zeros(qubit_state_num*mode_state_num, dtype=complex)
g[0] = 1

psi0 = [g]
print("starting states:")
print(g)

#Defining states to include in the drawing of occupation
states_draw_list = list(range(mode_state_num))
states_draw_names = []
for ii in range(mode_state_num):
    states_draw_names.append('g_' + str(ii))



#Defining convergence parameters
max_iterations = 500
decay = 250.0 # max_iterations/2 # amount by which convergence rate is suppressed exponentially in each iteration, smaller number is more suppression
# learning rate = (convergence rate) * exp(-(iteration #) / decay)
convergence = {'rate': 0.11, 'update_step': 5, 'max_iterations': max_iterations,
               'conv_target': 1e-4, 'learning_rate_decay': decay}



# Defining reg coeffs

states_forbidden_list = []

# for ii in range(mode_state_num):
#     forbid_state = (qubit_state_num-1)*mode_state_num+ii
#     if not forbid_state in states_forbidden_list:
#         states_forbidden_list.append(forbid_state)
#
# for ii in range(mode_state_num):
#     forbid_state = (qubit_state_num-2)*mode_state_num+ii
#     if not forbid_state in states_forbidden_list:
#         states_forbidden_list.append(forbid_state)

# for ii in range(mode_state_num):
#     forbid_state = (qubit_state_num-3)*mode_state_num+ii
#     if not forbid_state in states_forbidden_list:
#         states_forbidden_list.append(forbid_state)
#
# for ii in range(mode_state_num):
#     forbid_state = (qubit_state_num-4)*mode_state_num+ii
#     if not forbid_state in states_forbidden_list:
#         states_forbidden_list.append(forbid_state)

# for ii in range(mode_state_num):
#     forbid_state = (qubit_state_num-5)*mode_state_num+ii
#     if not forbid_state in states_forbidden_list:
#         states_forbidden_list.append(forbid_state)

# forbidding columns (higher energy states)
for ii in range(qubit_state_num):
    forbid_state = ii*mode_state_num + (mode_state_num-1)
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)

for ii in range(qubit_state_num):
    forbid_state = ii*mode_state_num + (mode_state_num-2)
    if not forbid_state in states_forbidden_list:
        states_forbidden_list.append(forbid_state)

#reg_coeffs = {'envelope' : 0.0, 'dwdt':0.00001,'d2wdt2':0.00001*0.0001, 'forbidden':100,
#             'states_forbidden_list': states_forbidden_list,'forbid_dressed':False,
#             'bandpass':1.0,'band':[0.1,10]}

states_forbidden_list = [qubit_state_num * mode_state_num - 1]
print(states_forbidden_list)

reg_coeffs = {'dwdt': 0.00007, 'forbidden_coeff_list': [10] * len(states_forbidden_list),
              'states_forbidden_list': states_forbidden_list, 'forbid_dressed': False}

initial_guess = None
initial_guess = np.ones((len(Hops), steps)) * 0.2
initial_guess = np.concatenate(([np.zeros(steps)], [np.zeros(steps)], [np.ones(steps)], [np.ones(steps)])) * 0.005

ss = Grape(H0, Hops, Hnames, U, total_time, steps, psi0, convergence=convergence,
                     draw=[states_draw_list, states_draw_names], state_transfer=state_transfer, use_gpu=False,
                     sparse_H=False, show_plots=False, Taylor_terms=[30,0], method='Adam', initial_guess=initial_guess,
                     maxA=ops_max_amp, reg_coeffs=reg_coeffs, dressed_info=dressed_info, file_name="binomial_fock",
                     data_path=data_path, LRF=False)
# Taylor_terms = [20, 0]: self.exp_terms, self.scaling