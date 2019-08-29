"""
binomialcode_exp1.py - Run binomial code experiment 1
"""

import os

import autograd.numpy as anp
import numpy as np
from qoc import (grape_schroedinger_discrete,)
from qoc.standard import (conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          matmuls,
                          TargetInfidelity,)

CORE_COUNT = 8
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define the system.
KAPPA = -2.26e-6 #GHz
ALPHA = -1.39e-1 #GHz
CHI_E = -5.61e-4 #GHz
CHI_F = -1.02e-3 #GHz
CHI_PRIME = 0
MAX_AMP_C = 2 * np.pi * 2e-3 #GHz
MAX_AMP_T = 2 * np.pi * 2e-2 #GHz

CAVITY_STATE_COUNT = 5
CAVITY_ZERO = np.array([[1], [0], [0], [0], [0]])
CAVITY_ONE = np.array([[0], [1], [0], [0], [0]])
CAVITY_TWO = np.array([[0], [0], [1], [0], [0]])
CAVITY_THREE = np.array([[0], [0], [0], [1], [0]])
CAVITY_FOUR = np.array([[0], [0], [0], [0], [1]])
CAVITY_PSI_UP = np.divide(CAVITY_ZERO + CAVITY_FOUR, np.sqrt(2))
CAVITY_PSI_DOWN = CAVITY_TWO
I_C = np.eye(CAVITY_STATE_COUNT)
A_C = get_annihilation_operator(CAVITY_STATE_COUNT)
A_DAGGER_C = get_creation_operator(CAVITY_STATE_COUNT)
A_DAGGER_2_A_2_C = matmuls(A_DAGGER_C, A_DAGGER_C, A_C, A_C)
N_C = matmuls(A_DAGGER_C, A_C)

TRANSMON_STATE_COUNT = 3
TRANSMON_G = np.array([[1], [0], [0]])
TRANSMON_E = np.array([[0], [1], [0]])
TRANSMON_F = np.array([[0], [0], [1]])
I_T = np.eye(TRANSMON_STATE_COUNT)
A_T = get_annihilation_operator(TRANSMON_STATE_COUNT)
A_DAGGER_T = get_creation_operator(TRANSMON_STATE_COUNT)
A_DAGGER_2_A_2_T = matmuls(A_DAGGER_T, A_DAGGER_T, A_T, A_T)
N_T = matmuls(A_DAGGER_T, A_T)

HILBERT_SIZE = CAVITY_STATE_COUNT * TRANSMON_STATE_COUNT
H_SYSTEM_0 = (np.divide(KAPPA, 2) * np.kron(A_DAGGER_2_A_2_C, I_T)
              + np.divide(ALPHA, 2) * np.kron(I_C, A_DAGGER_2_A_2_T)
              + CHI_E * np.kron(N_C, np.matmul(TRANSMON_E,
                                               conjugate_transpose(TRANSMON_E)))
              + CHI_F * np.kron(N_C, np.matmul(TRANSMON_F,
                                               conjugate_transpose(TRANSMON_F))))
              # + np.divide(CHI_PRIME, 2) * np.kron(A_DAGGER_2_A_2_C, N_T))
H_CONTROL_0_0 = np.kron(A_C, I_T)
H_CONTROL_0_1 = np.kron(A_DAGGER_C, I_T)
H_CONTROL_1_0 = np.kron(I_C, A_T)
H_CONTROL_1_1 = np.kron(I_C, A_DAGGER_T)
hamiltonian = (lambda params, t:
               H_SYSTEM_0
               + params[0] * H_CONTROL_0_0
               + anp.conjugate(params[0]) * H_CONTROL_0_1
               + params[1] * H_CONTROL_1_0
               + anp.conjugate(params[1]) * H_CONTROL_1_1)


# Define the problem.
INITIAL_STATE_0 = np.kron(CAVITY_ZERO, TRANSMON_G)
TARGET_STATE_0 = np.kron(CAVITY_PSI_UP, TRANSMON_G)
INITIAL_STATE_1 = np.kron(CAVITY_ZERO, TRANSMON_E)
TARGET_STATE_1 = np.kron(CAVITY_PSI_DOWN, TRANSMON_G)
INITIAL_STATES = np.stack((INITIAL_STATE_0, INITIAL_STATE_1,),)
TARGET_STATES = np.stack((TARGET_STATE_0, TARGET_STATE_1,),)
COSTS = (TargetInfidelity(TARGET_STATES),)

# Define the optimization.
PARAM_COUNT = 2
MAX_PARAM_NORMS = (MAX_AMP_C, MAX_AMP_T,)
PULSE_TIME = 500
PULSE_STEP_COUNT = 500
ITERATION_COUNT = 1000

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_FILE_NAME = "binomialcode_exp1"
SAVE_PATH = os.path.join("./pulses/", SAVE_FILE_NAME,)


def main():
    result = grape_schroedinger_discrete(COSTS, hamiltonian, INITIAL_STATES,
                                         ITERATION_COUNT, PARAM_COUNT,
                                         PULSE_STEP_COUNT, PULSE_TIME,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_param_norms=MAX_PARAM_NORMS,
                                         save_file_name=SAVE_FILE_NAME,
                                         save_iteration_step=SAVE_ITERATION_STEP,
                                         save_path=SAVE_PATH,)


if __name__ == "__main__":
    main()
