"""
mm_g0g4.py - This is a file to house the notebook QOC_test-Thoms.ipynb
created by Tanay.
"""


from argparse import ArgumentParser
from copy import copy
import os

import autograd.numpy as anp
from filelock import FileLock
import h5py
import numpy as np
from qutip import Qobj,tensor,basis
from qoc import (
    evolve_schroedinger_discrete,
    grape_schroedinger_discrete,
)
from qoc.standard import (
    Adam, LBFGSB,
    TargetStateInfidelity, ControlNorm, ControlVariation,
    conjugate_transpose, matmuls, krons,
    get_annihilation_operator, get_creation_operator,
    generate_save_file_path, plot_state_population,
    ForbidStates,
)

CORE_COUNT = 8
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define paths
EXPERIMENT_NAME = "mm_g0g4"
if "MULTIMODE_QOC_PATH" in os.environ:
    BASE_PATH = os.environ["MULTIMODE_QOC_PATH"]
else:
    BASE_PATH = "."
SAVE_PATH = os.path.join(BASE_PATH, "out", EXPERIMENT_NAME)

# Define experimental constants. All units are in GHz.
CAVITY_FREQ = 2 * np.pi * 4.4526
KAPPA = 2 * np.pi * -2.82e-6
TRANSMON_FREQ = 2 * np.pi * 5.6640
ALPHA = 2 * np.pi * -1.395126e-1
CHI_E = 2 * np.pi * -5.64453e-4
CHI_E_2 = 2 * np.pi * -7.3e-7
MAX_AMP_NORM_CAVITY = np.sqrt(2) * 2 * np.pi * 4e-4
MAX_AMP_NORM_TRANSMON = np.sqrt(2) * 2 * np.pi * 4e-3

# Define the system
CAVITY_STATE_COUNT = 3
CAVITY_ANNIHILATE = get_annihilation_operator(CAVITY_STATE_COUNT)
CAVITY_CREATE = get_creation_operator(CAVITY_STATE_COUNT)
CAVITY_NUMBER = np.matmul(CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_C2_A2 = matmuls(CAVITY_CREATE, CAVITY_CREATE, CAVITY_ANNIHILATE, CAVITY_ANNIHILATE)
CAVITY_ID = np.eye(CAVITY_STATE_COUNT)
CAVITY_VACUUM = np.zeros((CAVITY_STATE_COUNT, 1))
CAVITY_ZERO = np.copy(CAVITY_VACUUM)
CAVITY_ZERO[0][0] = 1
CAVITY_ONE = np.copy(CAVITY_VACUUM)
CAVITY_ONE[1][0] = 1
CAVITY_TWO = np.copy(CAVITY_VACUUM)
CAVITY_TWO[2][0] = 1

TRANSMON_STATE_COUNT = 3
TRANSMON_ANNIHILATE = get_annihilation_operator(TRANSMON_STATE_COUNT)
TRANSMON_CREATE = get_creation_operator(TRANSMON_STATE_COUNT)
TRANSMON_NUMBER = np.matmul(TRANSMON_CREATE, TRANSMON_ANNIHILATE)
TRANSMON_C2_A2 = matmuls(TRANSMON_CREATE, TRANSMON_CREATE, TRANSMON_ANNIHILATE, TRANSMON_ANNIHILATE)
TRANSMON_ID = np.eye(TRANSMON_STATE_COUNT)
TRANSMON_VACUUM = np.zeros((TRANSMON_STATE_COUNT, 1))
TRANSMON_G = np.copy(TRANSMON_VACUUM)
TRANSMON_G[0][0] = 1
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = np.copy(TRANSMON_VACUUM)
TRANSMON_E[1][0] = 1
TRANSMON_E_DAGGER = conjugate_transpose(TRANSMON_E)
TRANSMON_F = np.copy(TRANSMON_VACUUM)
TRANSMON_F[2][0] = 1
TRANSMON_F_DAGGER = conjugate_transpose(TRANSMON_F)

qnum = 2
mnum = 6

#Qubit rotation matrices
Q_x = np.diag(np.sqrt(np.arange(1, qnum)), 1)+np.diag(np.sqrt(np.arange(1, qnum)), -1)
Q_y = (0+1j) * (np.diag(np.sqrt(np.arange(1, qnum)), 1)-
                     np.diag(np.sqrt(np.arange(1, qnum)), -1))
Q_z = np.diag(np.arange(0, qnum))
I_q = np.identity(qnum)

#Cavity rotation matrices
M_x = np.diag(np.sqrt(np.arange(1, mnum)), 1)+np.diag(np.sqrt(np.arange(1, mnum)), -1)
M_y = (0+1j) * (np.diag(np.sqrt(np.arange(1, mnum)), 1)-
                     np.diag(np.sqrt(np.arange(1, mnum)), -1))
M_z = np.diag(np.arange(0, mnum))
I_m = np.identity(mnum)

am =  Qobj(np.kron(I_q, np.diag(np.sqrt(np.arange(1, mnum)), 1))) #tensor product of the qubit
#identity with anhilation of the cavity state
aq =  Qobj(np.kron(np.diag(np.sqrt(np.arange(1, qnum)), 1), I_m ))
sigmaz_q = Qobj(np.kron(Q_z, I_m)) #z operator on the qubit

mode = 2 #choose a particular mode of the MM cavity
if mode == 0: 
    chi_e = -1.19387842e-3
elif mode == 2:
    chi_e = -0.663547253e-3
elif mode == 4:
    chi_e = -0.465719362e-3



hparams = {"chi":[chi_e, -0.9147204246769476e-3],"kappa":5.23e-6,"omega":0.144e-3, "chi_2":[0.001e-3, 2*0.001e-3 ]}
def H_rot():
        
    chi, kappa, omega, chi_2 = hparams["chi"], hparams["kappa"],\
                                   hparams["omega"],  hparams["chi_2"]
    freq_ge, mode_ens = 0, 0 # GHz, in lab frame
    chi_mat = np.zeros(qnum)
    chi_2_mat = np.zeros(qnum)

    if qnum <= 2:
        chi_mat[1] = chi[0] # ge
        chi_2_mat[1] = chi_2[0] #ge

    if qnum > 2:
        chi_mat[1] = chi[0] # ge
        chi_mat[2] = chi[1] #ef
        chi_2_mat[1] = chi_2[0] #ge
        chi_2_mat[2] = chi_2[1] #ef

    #self-kerr of the cavity modes    
    mode_freq = 0
    mode_ens = np.array([2*np.pi*ii*(mode_freq - 0.5*(ii-1)*kappa) for ii in np.arange(mnum)])
    H_m = np.diag(mode_ens)

    H0_1 = np.kron(I_q, H_m) + 2 * 2 * np.pi * (np.kron(np.diag(chi_mat), M_z))
    H0_2 =2*np.pi*np.kron(np.diag(chi_2_mat), np.diag(np.array([ii*(ii-1) for ii in np.arange(mnum)])))

#     if self.use_full_H:
    return (H0_1 + H0_2)

H_SYSTEM = H_rot()

XI = np.kron(Q_x, I_m)
YI = np.kron(Q_y, I_m)
IX = np.kron(I_q, M_x)
IY = np.kron(I_q, M_y)
Hops = []
Hops.extend([XI, YI, IX, IY]) 


hamiltonian = lambda controls, time: (
    H_SYSTEM
    + controls[0] * Hops[0] + controls[1] * Hops[1] + controls[2] * Hops[2] + controls[3] * Hops[3]
)
CONTROL_COUNT = 4
COMPLEX_CONTROLS = False
MAX_CONTROL_NORMS = np.array((MAX_AMP_NORM_TRANSMON,MAX_AMP_NORM_TRANSMON,\
                              MAX_AMP_NORM_CAVITY,MAX_AMP_NORM_CAVITY))


# Define the problem
EVOLUTION_TIME = 2e3 #ns
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME) + 1
psi0 = tensor(basis(qnum,0),basis(mnum,0)).full()
psi1 = tensor(basis(qnum,0),basis(mnum,4)).full()
INITIAL_STATES = np.stack((psi0,))
TARGET_STATES = np.stack((psi1,))
FORBIDDEN_STATES = np.stack([[psi0, psi1]])
COSTS = [
    TargetStateInfidelity(TARGET_STATES), 
    ControlVariation(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                     cost_multiplier=2, max_control_norms=MAX_CONTROL_NORMS,
                     order=1),
    ControlVariation(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                     cost_multiplier=2, max_control_norms=MAX_CONTROL_NORMS,
                     order=2),
    ForbidStates(FORBIDDEN_STATES, SYSTEM_EVAL_COUNT),
]

# Define the optimization
LEARNING_RATE = 1e-5
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
ITERATION_COUNT = int(3e3)
GRAB_CONTROLS = True
if GRAB_CONTROLS:
    controls_path = os.path.join(SAVE_PATH, "00003_mm_g0g4.h5")
    controls_path_lock = "{}.lock".format(controls_path)
    with FileLock(controls_path_lock):
        with h5py.File(controls_path) as save_file:
            index = np.argmin(save_file["error"])
            controls = save_file["controls"][index][()]
        #ENDWITH
    #ENDWITH
    INITIAL_CONTROLS = controls
else:
    INITIAL_CONTROLS = None


def impose_control_conditions(controls):
    # Impose zero at boundaries
    controls[0, :] = controls[-1, :] = 0
    return controls


# Define the output.
LOG_ITERATION_STEP = 5
SAVE_ITERATION_STEP = 20
SAVE_INTERMEDIATE_STATES = True

GRAPE_CONFIG = {
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "complex_controls": COMPLEX_CONTROLS,
    "initial_controls": INITIAL_CONTROLS,
    "impose_control_conditions": impose_control_conditions,
    "iteration_count": ITERATION_COUNT,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_control_norms": MAX_CONTROL_NORMS,
    "optimizer": OPTIMIZER,
    "save_intermediate_states" : SAVE_INTERMEDIATE_STATES,
    "save_iteration_step": SAVE_ITERATION_STEP,
}

EVOL_CONFIG = {
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
}

def run_grape():
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)


def run_evolve():
    result = evolve_schroedinger_discrete(**EVOL_CONFIG)
    print(result.error)


def main():
    parser = ArgumentParser()
    parser.add_argument("--grape", action="store_true")
    parser.add_argument("--evolve", action="store_true")
    args = vars(parser.parse_args())
    do_grape = args["grape"]
    do_evolve = args["evolve"]

    if do_grape:
        run_grape()
    elif do_evolve:
        run_evolve()

if __name__ == "__main__":
    main()
