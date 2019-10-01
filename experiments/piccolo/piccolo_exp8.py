"""
piccolo_exp8.py - Experiment 8 for the piccolo system.
"""

import os

import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (conjugate_transpose,
                          matmuls,
                          get_annihilation_operator,
                          get_creation_operator,
                          TargetStateInfidelity,
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
MAX_AMP_C = 2 * anp.pi * 2e-3 #GHz
MAX_AMP_T = 2 * anp.pi * 2e-2 #GHz

# Define the system.
CAVITY_STATE_COUNT = 20
CAVITY_ANNIHILATE = get_annihilation_operator(CAVITY_STATE_COUNT)
CAVITY_CREATE = get_creation_operator(CAVITY_STATE_COUNT)
CAVITY_NUMBER = anp.matmul(CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_QUADRATURE = matmuls(CAVITY_CREATE, CAVITY_CREATE,
                            CAVITY_ANNIHILATE, CAVITY_ANNIHILATE)
CAVITY_I = anp.eye(CAVITY_STATE_COUNT)
CAVITY_VACUUM = anp.zeros((CAVITY_STATE_COUNT, CAVITY_STATE_COUNT))
CAVITY_ZERO = anp.copy(CAVITY_VACUUM)
CAVITY_ZERO[0][0] = 1.
CAVITY_ONE = anp.copy(CAVITY_VACUUM)
CAVITY_ONE[1][1] = 1.

TRANSMON_STATE_COUNT = 2
TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, TRANSMON_STATE_COUNT))
TRANSMON_G = anp.copy(TRANSMON_VACUUM)
TRANSMON_G[0][0] = 1.
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = anp.copy(TRANSMON_VACUUM)
TRANSMON_E[1][1] = 1.
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

MAX_PARAM_NORMS = anp.array((MAX_AMP_T, MAX_AMP_C,))
COMPLEX_CONTROLS = True
CONTROL_COUNT = 2
EVOLUTION_TIME = 500 #ns
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME)

# Define the optimization parameters.
ITERATION_COUNT = 5000
OPTIMIZER = Adam()

# Define the problem.
INITIAL_STATE_0 = anp.kron(CAVITY_ZERO, TRANSMON_G)
INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
TARGET_STATE_0 = anp.kron(CAVITY_ONE, TRANSMON_G)
TARGET_STATES = anp.stack((TARGET_STATE_0,))
COSTS = (TargetStateInfidelity(TARGET_STATES),
         # ParamVariation(MAX_PARAM_NORMS,
         #                PARAM_COUNT,
         #                PULSE_STEP_COUNT,
         #                cost_multiplier=0.5,
         #                order=1),
         # ParamVariation(MAX_PARAM_NORMS,
         #                PARAM_COUNT,
         #                PULSE_STEP_COUNT,
         #                cost_multiplier=0.5,
         #                order=2),
)

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_FILE_NAME = "piccolo_exp8"
SAVE_ITERATION_STEP = 1
SAVE_PATH = os.path.join("./pulses/", SAVE_FILE_NAME,)
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH)


def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                         COSTS, EVOLUTION_TIME, hamiltonian,
                                         INITIAL_STATES, SYSTEM_EVAL_COUNT,
                                         complex_controls=COMPLEX_CONTROLS,
                                         iteration_count=ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_control_norms=MAX_PARAM_NORMS,
                                         optimizer=OPTIMIZER,
                                         save_file_path=SAVE_FILE_PATH,)


if __name__ == "__main__":
    main()
