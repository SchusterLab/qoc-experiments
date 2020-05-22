import argparse
import os, time, scipy, ntpath, h5py, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from filelock import FileLock, Timeout
from copy import copy
from numpy import pi
from qutip import Qobj, tensor, basis, fock, num, qeye, destroy, fidelity, expect, ket2dm, mesolve

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

#=== QOC package imports ===#
from qoc import (
    evolve_schroedinger_discrete,
    grape_schroedinger_discrete,
)
from qoc.standard import (
    Adam, LBFGSB,
    TargetStateInfidelity, ControlNorm, ControlVariation, ForbidStates, ControlArea,
    conjugate_transpose, matmuls, krons, matrix_to_column_vector_list,
    get_annihilation_operator, get_creation_operator,
    generate_save_file_path,
)

def create_swap_unitary(psi1, psi2):
    psi1 = Qobj(psi1).unit()

    psi2 = Qobj(psi2).unit()
    op = qeye(psi1.dims[0]) - psi1 * psi1.dag() - psi2 * psi2.dag()
    op_ = op + psi1 * psi2.dag() + psi2 * psi1.dag()
    # print("psi1:\n{}\npsi2:\n{}"
    #       "".format(psi1, psi2))
    return op_

CORE_COUNT = 8
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define paths
EXPERIMENT_NAME = "swap_g10_g01"
EXPERIMENT_META = "swap_g10_g01"

qnum = 2
mnum_a = 4
mnum_b = 4
# matrix_to_column_vector_list
psi1 = tensor(basis(qnum,0), basis(mnum_a,1) ,basis(mnum_b,0))
psi2 = tensor(basis(qnum,0), basis(mnum_a,0) ,basis(mnum_b,1))
target_unitary = create_swap_unitary(psi1,psi2)


# Define experimental constants. All units are in GHz.
max_amp_qubit  = 2*np.pi * 7.5e-3
max_amp_mode_a = 2*np.pi * 1.5e-3
max_amp_mode_b = 2*np.pi * 1.5e-3

# Define the optimization
EVOLUTION_TIME = 2e3 #ns
LEARNING_RATE = 5e-4
ITERATION_COUNT = int(2e3)
# Define the output.
LOG_ITERATION_STEP = 2
SAVE_ITERATION_STEP = 50

chi_e_m0 = -1.19387842e-3
chi_e_m2 = -0.663547253e-3 # mode = 2
chi_e_m4 = -0.465719362e-3

hparams = {"chi_a":[chi_e_m0, -0.9147204246769476e-3], "chi_b":[chi_e_m2, -0.9147204246769476e-3], \
           "chi2_a":[0.001e-3, 2*0.001e-3 ], "chi2_b":[0.001e-3, 2*0.001e-3 ], \
           "chi_ab": -10e-6, "chi2_ab": -1e-6, \
           "K_a":-5.23e-6, "K_b":-5.20e-6} # In linear frequency

#Qubit rotation matrices
a = destroy(qnum)
Q_x = (a + a.dag()).full()
Q_y = -1j* (a - a.dag()).full()
Q_z = np.diag(np.arange(0, qnum))
Q_I = np.identity(qnum)

#Cavity mode A rotation matrices
a = destroy(mnum_a)
M_x_a = (a + a.dag()).full()
M_y_a = -1j* (a - a.dag()).full()
M_z_a = np.diag(np.arange(0, mnum_a))
M_I_a = np.identity(mnum_a)

#Cavity mode B rotation matrices
a = destroy(mnum_b)
M_x_b = (a + a.dag()).full()
M_y_b = -1j* (a - a.dag()).full()
M_z_b = np.diag(np.arange(0, mnum_b))
M_I_b = np.identity(mnum_b)


def H_rot():
    chi_a, chi_b, chi2_a, chi2_b = hparams["chi_a"], hparams["chi_b"], hparams["chi2_a"], hparams["chi2_b"]
    chi_ab, chi2_ab, K_a, K_b =    hparams["chi_ab"], hparams["chi2_ab"], hparams["K_a"], hparams["K_b"]
                            
    freq_ge, mode_ens = 0, 0 # GHz, in lab frame
    chi_a_arr = np.zeros(qnum)
    chi_b_arr = np.zeros(qnum)
    chi2_a_arr = np.zeros(qnum)
    chi2_b_arr = np.zeros(qnum)

    chi_a_arr[1] = chi_a[0] # ge of a
    chi_b_arr[1] = chi_b[0] # ge of b
    chi2_a_arr[1] = chi2_a[0] #ge of a
    chi2_b_arr[1] = chi2_b[0] #ge of b

    if qnum > 2:
        chi_a_arr[2] = chi_a[1] #ef of a 
        chi_b_arr[2] = chi_b[1] #ef of b
        chi2_a_arr[2] = chi2_a[1] #ef of a
        chi2_b_arr[2] = chi2_b[1] #ef of b

    #self-kerr of the cavity modes    
    mode_ens_a = np.array([K_a/2 *ii*(ii-1) for ii in np.arange(mnum_a)])
    mode_ens_b = np.array([K_b/2 *ii*(ii-1) for ii in np.arange(mnum_b)])
    H_m_a = np.diag(mode_ens_a)
    H_m_b = np.diag(mode_ens_b)

    H0 = krons(Q_I, H_m_a, M_I_b) + krons(Q_I, M_I_a, H_m_b)
    H0_1 = 2 * (krons(np.diag(chi_a_arr), M_z_a, M_I_b) + krons(np.diag(chi_b_arr), M_I_a, M_z_b))
    H0_2 = krons(np.diag(chi2_a_arr), np.diag(np.array([ii*(ii-1) for ii in np.arange(mnum_a)])), M_I_b)\
         + krons(np.diag(chi2_b_arr), M_I_a, np.diag(np.array([ii*(ii-1) for ii in np.arange(mnum_b)])))
    H0_3 = 2 *chi_ab * krons(Q_I, num(mnum_a).full(), num(mnum_b).full()) \
         + 0*chi2_ab * krons(num(qnum).full(), num(mnum_a).full(), num(mnum_b).full()) # Check this line with Vatsan

    return 2*pi*(H0 + H0_1 + H0_2)

H_SYSTEM = H_rot()

XII = krons(Q_x, M_I_a, M_I_b)
YII = krons(Q_y, M_I_a, M_I_b)
IXI = krons(Q_I, M_x_a, M_I_b)
IYI = krons(Q_I, M_y_a, M_I_b)
IIX = krons(Q_I, M_I_a, M_x_b)
IIY = krons(Q_I, M_I_a, M_y_b)
Hops = [XII, YII, IXI, IYI, IIX, IIY]

CONTROL_COUNT = len(Hops)
COMPLEX_CONTROLS = False
MAX_CONTROL_NORMS = np.array((max_amp_qubit,max_amp_qubit,\
                              max_amp_mode_a,max_amp_mode_a,max_amp_mode_b,max_amp_mode_b))

hamiltonian = lambda controls, time: (
    H_SYSTEM + controls[0] * Hops[0] + controls[1] * Hops[1] 
             + controls[2] * Hops[2] + controls[3] * Hops[3]
             + controls[4] * Hops[4] + controls[5] * Hops[5]
)

# def hamiltonian(controls, time):
#     return (H_SYSTEM + controls[0] * Hops[0] + controls[1] * Hops[1] 
#                      + controls[2] * Hops[2] + controls[3] * Hops[3])

# def hamiltonian(controls, time):
#     Htemp = 0
#     for i in range(CONTROL_COUNT):
#         Htemp += controls[i] * Hops[i]
#     return (H_SYSTEM + Htemp)

# Define the problem
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME) + 1
hilbert_size = qnum * mnum_a * mnum_b
target_unitary_ = target_unitary.full()

INITIAL_STATES = matrix_to_column_vector_list(np.identity(hilbert_size))
TARGET_STATES = matrix_to_column_vector_list(target_unitary_)

# INITIAL_STATES = np.stack([psi1, psi2])
# TARGET_STATES = np.stack([psi2, psi1])

# For Gaussian envelop
def gauss(x):
    b = anp.mean(x)
    c = anp.std(x)
    return anp.exp(-((x - b) ** 2) / (c ** 2))
CONTROL_NORM_WEIGHTS = 1 - gauss(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT))
CONTROL_NORM_WEIGHTS = anp.repeat(CONTROL_NORM_WEIGHTS[:, anp.newaxis], CONTROL_COUNT, axis=1)
# COSTS = [
# ControlNorm(CONTROL_COUNT, CONTROL_EVAL_COUNT,
#                      control_weights=CONTROL_NORM_WEIGHTS,
#                      cost_multiplier=1,
#                      max_control_norms=MAX_CONTROL_NORMS,),
# ]


# psi_f1 = tensor(basis(qnum,0),basis(mnum,11)).full()
# psi_f2 = tensor(basis(qnum,1),basis(mnum,11)).full()
# forbidden_states = np.stack([[psi_f1, psi_f2] for _ in range(STATE_COUNT)]) # For multiple states
# forbidden_states = np.stack([[psi_f1, psi_f2]],)

COSTS = [
    TargetStateInfidelity(TARGET_STATES),
    # ControlVariation(CONTROL_COUNT, CONTROL_EVAL_COUNT,
    #                  cost_multiplier=1, max_control_norms=MAX_CONTROL_NORMS, order=1),
    # ControlVariation(CONTROL_COUNT, CONTROL_EVAL_COUNT,
    #                  cost_multiplier=1, max_control_norms=MAX_CONTROL_NORMS, order=2),
]
# ForbidStates(forbidden_states, SYSTEM_EVAL_COUNT, cost_multiplier=10),
# ControlArea(CONTROL_COUNT, CONTROL_EVAL_COUNT, cost_multiplier=1, max_control_norms=MAX_CONTROL_NORMS),

WDIR = os.environ["MULTIMODE_QOC_PATH"]
SAVE_PATH = os.path.join(WDIR, "out", EXPERIMENT_META)
                
GRAB_CONTROLS = True
if GRAB_CONTROLS:
    controls_path = os.path.join(SAVE_PATH, "00002_swap_g10_g01.h5")
    controls_path_lock = "{}.lock".format(controls_path)
    with FileLock(controls_path_lock):
        with h5py.File(controls_path) as save_file:
            index = np.argmin(save_file["error"])
            controls = save_file["controls"][index][()]
        #ENDWITH
    #ENDWITH
    INITIAL_CONTROLS = controls
else:
    tarr = np.linspace(0,2*np.pi,CONTROL_EVAL_COUNT)
    cos_arr = 0.25*np.cos(6*tarr)
    sin_arr = 0.25*np.sin(6*tarr)
    INITIAL_CONTROLS = np.stack((max_amp_qubit*sin_arr,max_amp_qubit*sin_arr,
                                 max_amp_mode_a*sin_arr,max_amp_mode_a*sin_arr,
                                 max_amp_mode_b*sin_arr,max_amp_mode_b*sin_arr), axis=1)
    
    INITIAL_CONTROLS = None
# ============ #


OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

SAVE_INTERMEDIATE_STATES = True

def impose_control_conditions(controls):
    # Impose zero at boundaries
    controls[0, :] = controls[-1, :] = 0
    return controls

GRAPE_CONFIG = {
    "control_count": CONTROL_COUNT,
    "control_eval_count": CONTROL_EVAL_COUNT,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "complex_controls": COMPLEX_CONTROLS,
    "impose_control_conditions": impose_control_conditions,
    "initial_controls": INITIAL_CONTROLS,
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
    t_start = time.process_time()
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path
    })
    result = grape_schroedinger_discrete(**config)
    
    tot_time = time.process_time() - t_start
    print("Total time (%s iterations): %d s, time/ieration: %.2f s" 
          %(ITERATION_COUNT, tot_time, tot_time/ITERATION_COUNT))


def run_evolve():
    result = evolve_schroedinger_discrete(**EVOL_CONFIG)
    print(result.error)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--grape", action="store_true")
    args = vars(parser.parse_args())
    do_grape = args["grape"]
    do_evolve = args["evolve"]

    if do_grape:
        run_grape()

    if do_evolve:
        run_evolve()


if __name__ == "__main__":
    main()
