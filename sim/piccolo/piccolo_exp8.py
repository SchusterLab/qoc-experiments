"""
piccolo_exp8.py - Experiment 8 for the piccolo system.
"""

import os

import autograd.numpy as anp
from filelock import FileLock, Timeout
import h5py
from qocstable import grape_schroedinger_discrete
from qocstable.standard import (conjugate_transpose,
                                 matmuls,
                                 get_annihilation_operator,
                                 get_creation_operator,
                                 TargetStateInfidelity,
                                 ControlNorm,
                                 ControlVariation,
                                 TargetStateInfidelityTime,
                                 LBFGSB, Adam,
                                 generate_save_file_path,)

# Specify computer specs.
CORE_COUNT = 8
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define experimental constants.
CHI_E = -5.65e-4 #GHz
CHI_E_2 = 7.3e-7
KAPPA = 2.09e-6 #GHz
MAX_AMP_C = 2 * anp.pi * 2e-4 #GHz
MAX_AMP_T = 2 * anp.pi * 3e-3 #GHz

# Define the system.
CAVITY_STATE_COUNT = 3
CAVITY_ANNIHILATE = get_annihilation_operator(CAVITY_STATE_COUNT)
CAVITY_CREATE = get_creation_operator(CAVITY_STATE_COUNT)
CAVITY_NUMBER = anp.matmul(CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_QUADRATURE = matmuls(CAVITY_CREATE, CAVITY_CREATE,
                            CAVITY_ANNIHILATE, CAVITY_ANNIHILATE)
CAVITY_I = anp.eye(CAVITY_STATE_COUNT)
CAVITY_VACUUM = anp.zeros((CAVITY_STATE_COUNT, 1))
CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
CAVITY_ZERO[0][0] = 1.
CAVITY_ONE = anp.copy(CAVITY_VACUUM)
CAVITY_ONE[1][0] = 1.

TRANSMON_STATE_COUNT = 2
TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, 1))
TRANSMON_G = anp.copy(TRANSMON_VACUUM)
TRANSMON_G[0][0] = 1.
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = anp.copy(TRANSMON_VACUUM)
TRANSMON_E[1][0] = 1.
TRANSMON_E_DAGGER = conjugate_transpose(TRANSMON_E)
TRANSMON_I = anp.eye(TRANSMON_STATE_COUNT)

H_SYSTEM = (2 * CHI_E * anp.kron(CAVITY_NUMBER, anp.matmul(TRANSMON_E, TRANSMON_E_DAGGER))
            + CHI_E_2 * anp.kron(CAVITY_QUADRATURE, anp.matmul(TRANSMON_E, TRANSMON_E_DAGGER))
            + KAPPA / 2 * anp.kron(CAVITY_QUADRATURE, TRANSMON_I))
H_GE = anp.kron(CAVITY_I, anp.matmul(TRANSMON_G, TRANSMON_E_DAGGER))
H_GE_DAGGER = conjugate_transpose(H_GE)
H_C = anp.kron(CAVITY_ANNIHILATE, TRANSMON_I)
H_C_DAGGER = conjugate_transpose(H_C)
hamiltonian = (lambda controls, time:
               H_SYSTEM
               + controls[0] * H_GE
               + anp.conjugate(controls[0]) * H_GE_DAGGER
               + controls[1] * H_C
               + anp.conjugate(controls[1]) * H_C_DAGGER)

MAX_CONTROL_NORMS = anp.array((MAX_AMP_T, MAX_AMP_C,))
COMPLEX_CONTROLS = True
CONTROL_COUNT = 2
EVOLUTION_TIME = int(3e3) #ns
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME)

# Define the optimization parameters.
ITERATION_COUNT = int(1e4)
OPTIMIZER = LBFGSB()
# FILE_PATH = "./pulses/piccolo_exp8/00035_piccolo_exp8.h5"
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
INITIAL_CONTROLS = (anp.ones((CONTROL_EVAL_COUNT, CONTROL_COUNT), dtype=anp.complex128)
                    * MAX_CONTROL_NORMS
                    / anp.sqrt(2))
# def impose_control_conditions(controls):
#     """
#     Impose 0 on the control boundaries.
#     """
#     controls[0,:]= 0
#     controls[-1, :] = 0
#     return controls
impose_control_conditions = None

# Define the problem.
INITIAL_STATE_0 = anp.kron(CAVITY_ZERO, TRANSMON_G)
INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
TARGET_STATE_0 = anp.kron(CAVITY_ONE, TRANSMON_G)
TARGET_STATES = anp.stack((TARGET_STATE_0,))
FIDELITY_MULTIPLIER = 1.
CONTROL_NORM_MULTIPLIER = 1.
def gauss(x):
    b = anp.mean(x)
    c = anp.std(x)
    return anp.exp(-((x - b) ** 2) / (c ** 2))
CONTROL_NORM_WEIGHTS = 1 - gauss(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT))
CONTROL_NORM_WEIGHTS = anp.repeat(CONTROL_NORM_WEIGHTS[:, anp.newaxis], CONTROL_COUNT, axis=1)
# Don't penalize controls in the middle.
CONTROL_OFFSET = 200
CONTROL_NORM_WEIGHTS[CONTROL_OFFSET:-CONTROL_OFFSET] = 0
CONTROL_VAR_O1_MULTIPLIER = 1.
CONTROL_VAR_O2_MULTIPLIER = 1.
COSTS = (TargetStateInfidelity(TARGET_STATES,
                               cost_multiplier=FIDELITY_MULTIPLIER),
         # ControlNorm(CONTROL_COUNT, CONTROL_EVAL_COUNT,
         #             control_weights=CONTROL_NORM_WEIGHTS,
         #             cost_multiplier=CONTROL_NORM_MULTIPLIER,
         #             max_control_norms=MAX_CONTROL_NORMS,),
         # ControlVariation(CONTROL_COUNT,
         #                  CONTROL_EVAL_COUNT,
         #                  cost_multiplier=CONTROL_VAR_O1_MULTIPLIER,
         #                  max_control_norms=MAX_CONTROL_NORMS,
         #                  order=1),
         # ControlVariation(CONTROL_COUNT,
         #                  CONTROL_EVAL_COUNT,
         #                  cost_multiplier=CONTROL_VAR_O2_MULTIPLIER,
         #                  max_control_norms=MAX_CONTROL_NORMS,
         #                  order=2),
)

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_FILE_NAME = "piccolo_exp8"
SAVE_ITERATION_STEP = 1
SAVE_PATH = os.path.join("./pulses/", SAVE_FILE_NAME,)
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)
SAVE_INTERMEDIATE_STATES = False

def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                         complex_controls=COMPLEX_CONTROLS,
                                         impose_control_conditions=impose_control_conditions,
                                         initial_controls=INITIAL_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_control_norms=MAX_CONTROL_NORMS,
                                         optimizer=OPTIMIZER,
                                         save_file_path=SAVE_FILE_PATH,
                                         save_intermediate_states=SAVE_INTERMEDIATE_STATES,
                                         save_iteration_step=SAVE_ITERATION_STEP,
    )


if __name__ == "__main__":
    main()
