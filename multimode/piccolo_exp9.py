"""
piccolo_exp9.py - Experiment 9 for the piccolo system.
"""

import os

import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
from qoc import (grape_schroedinger_discrete,
                 evolve_schroedinger_discrete)
from qoc.standard import (conjugate_transpose,
                          matmuls,
                          get_annihilation_operator,
                          get_creation_operator,
                          TargetStateInfidelity,
                          ControlBandwidthMax,
                          ControlNorm,
                          ControlVariation,
                          TargetStateInfidelityTime,
                          LBFGSB, Adam,
                          generate_save_file_path,)

# Paths.
DATA_PATH = os.path.join(os.environ["MULTIMODE_QOC_PATH"], "out")

# Specify computer specs.
CORE_COUNT = 8

# Define experimental constants.
BLOCKADE_LEVEL = 3
CHI_E = 2 * anp.pi * -5.672016e-4 #GHz
KAPPA = 2 * anp.pi * 2.09e-6 #GHz
OMEGA = 2 * anp.pi * 1.44e-4 #GHz
MAX_AMP_C = anp.sqrt(2) * 2 * anp.pi * 1.5e-5 #GHz
MAX_BND_C = 2 * OMEGA

# Define the system.
CAVITY_STATE_COUNT = 5
CAVITY_ANNIHILATE = get_annihilation_operator(CAVITY_STATE_COUNT)
CAVITY_CREATE = get_creation_operator(CAVITY_STATE_COUNT)
CAVITY_NUMBER = anp.matmul(CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_QUADRATURE = matmuls(CAVITY_CREATE, CAVITY_ANNIHILATE, CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_I = anp.eye(CAVITY_STATE_COUNT)
CAVITY_VACUUM = anp.zeros((CAVITY_STATE_COUNT, 1))
CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
CAVITY_ZERO[0][0] = 1.
CAVITY_ONE = anp.copy(CAVITY_VACUUM)
CAVITY_ONE[1][0] = 1.

TRANSMON_STATE_COUNT = 2
TRANSMON_I = anp.eye(TRANSMON_STATE_COUNT)
TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, 1))
TRANSMON_G = anp.copy(TRANSMON_VACUUM)
TRANSMON_G[0][0] = 1.
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = anp.copy(TRANSMON_VACUUM)
TRANSMON_E[1][0] = 1.
TRANSMON_E_DAGGER = conjugate_transpose(TRANSMON_E)

H_SYSTEM = (2 * CHI_E * anp.kron(CAVITY_NUMBER - BLOCKADE_LEVEL, anp.matmul(TRANSMON_E, TRANSMON_E_DAGGER))
            + OMEGA * anp.kron(CAVITY_I,
                               anp.matmul(TRANSMON_G, TRANSMON_E_DAGGER)
                               + anp.matmul(TRANSMON_E, TRANSMON_G_DAGGER))
            + KAPPA / 2 * anp.kron(anp.matmul(CAVITY_NUMBER, CAVITY_NUMBER - CAVITY_I), TRANSMON_I))
assert(anp.allclose(H_SYSTEM, conjugate_transpose(H_SYSTEM)))
H_C = anp.kron(CAVITY_ANNIHILATE, TRANSMON_I)
H_C_DAGGER = conjugate_transpose(H_C)
hamiltonian = (lambda controls, time:
               H_SYSTEM
               + controls[0] * H_C
               + anp.conjugate(controls[0]) * H_C_DAGGER)

MAX_CONTROL_NORMS = anp.array((MAX_AMP_C,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_BND_C,))
COMPLEX_CONTROLS = True
CONTROL_COUNT = 1
EVOLUTION_TIME = int(2.5e4) #ns
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME)

# Define the optimization parameters.
ITERATION_COUNT = int(1e4)
LEARNING_RATE = 1e-5
OPTIMIZER = LBFGSB()
# Set initial controls to optimal controls found by another iteration.
# FILE_PATH = os.path.join(DATA_PATH, "piccolo_exp9/00005_piccolo_exp9.h5")
# FILE_LOCK_PATH = "{}.lock".format(FILE_PATH)
# try:
#     with FileLock(FILE_LOCK_PATH):
#         with h5py.File(FILE_PATH) as file_:
#             index = anp.argmin(file_["error"])
#             controls = file_["controls"][index]
# except Timeout:
#     print("Timeout encountered")
#     exit(0)
# INITIAL_CONTROLS = controls
INITIAL_CONTROLS = None
def impose_control_conditions(controls):
    """
    Impose 0 on the control boundaries.
    """
    controls[0,:]= 0
    controls[-1, :] = 0
    return controls
# impose_control_conditions = None

# Define the problem.
INITIAL_STATE_0 = anp.kron(CAVITY_ZERO, TRANSMON_G)
INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
TARGET_STATE_0 = anp.kron(CAVITY_ONE, TRANSMON_G)
TARGET_STATES = anp.stack((TARGET_STATE_0,))
def gauss(x):
    b = anp.mean(x)
    c = anp.std(x)
    return anp.exp(-((x - b) ** 2) / (c ** 2))
CONTROL_NORM_WEIGHTS = 1 - gauss(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT))
CONTROL_NORM_WEIGHTS = anp.repeat(CONTROL_NORM_WEIGHTS[:, anp.newaxis], CONTROL_COUNT, axis=1)
# Don't penalize controls in the middle.
# CONTROL_OFFSET = int(CONTROL_COUNT / 5)
# CONTROL_NORM_WEIGHTS[CONTROL_OFFSET:-CONTROL_OFFSET] = 0
CONTROL_BANDWIDTH_MULTIPLIER = 1.
CONTROL_NORM_MULTIPLIER = 0.5
CONTROL_VAR_01_MULTIPLIER = 1.
CONTROL_VAR_02_MULTIPLIER = 1.
FIDELITY_MULTIPLIER = 1.
COSTS = [
    TargetStateInfidelity(TARGET_STATES,
                          cost_multiplier=FIDELITY_MULTIPLIER),
     ControlVariation(CONTROL_COUNT,
                      CONTROL_EVAL_COUNT,
                      cost_multiplier=CONTROL_VAR_01_MULTIPLIER,
                      max_control_norms=MAX_CONTROL_NORMS,
                      order=1),
     ControlVariation(CONTROL_COUNT,
                      CONTROL_EVAL_COUNT,
                      cost_multiplier=CONTROL_VAR_02_MULTIPLIER,
                      max_control_norms=MAX_CONTROL_NORMS,
                      order=2),
     # ControlNorm(CONTROL_COUNT, CONTROL_EVAL_COUNT,
     #             control_weights=CONTROL_NORM_WEIGHTS,
     #             cost_multiplier=CONTROL_NORM_MULTIPLIER,
     #             max_control_norms=MAX_CONTROL_NORMS,),
     # ControlBandwidthMax(CONTROL_COUNT, CONTROL_EVAL_COUNT, EVOLUTION_TIME,
     #                     MAX_CONTROL_BANDWIDTHS,
     #                     cost_multiplier=CONTROL_BANDWIDTH_MULTIPLIER),
]

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_FILE_NAME = "piccolo_exp9"
SAVE_ITERATION_STEP = 1
SAVE_PATH = os.path.join(DATA_PATH, SAVE_FILE_NAME,)
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
SAVE_INTERMEDIATE_STATES = False

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
    "save_file_path": SAVE_FILE_PATH,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES,
    "save_iteration_step": SAVE_ITERATION_STEP,
}

EVOL_CONFIG = {
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
}

def main():
    result = grape_schroedinger_discrete(**GRAPE_CONFIG)
    # result = evolve_schroedinger_discrete(**EVOL_CONFIG)
    # print(result.error)


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
    os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
    main()
