"""
piccolo_exp7.py - experiment 7 for the piccolo system.
"""

import os

import autograd.numpy as anp
import h5py
from qoc import grape_schroedinger_discrete
from qoc.standard import (conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          TargetStateInfidelity,
                          ControlVariation,
                          LBFGSB, Adam,
                          generate_save_file_path,)

# Specify computer specs.
CORE_COUNT = 8
os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)

# Define experimental constants.
CHI_E = -5.65e-4 #GHz
CHI_F = 2 * CHI_E #GHz
MAX_AMP_C = 2 * anp.pi * 2e-3 #GHz
MAX_AMP_T = 2 * anp.pi * 2e-2 #GHz

# Define the system.
CAVITY_STATE_COUNT = 5
CAVITY_ANNIHILATE = get_annihilation_operator(CAVITY_STATE_COUNT)
CAVITY_CREATE = get_creation_operator(CAVITY_STATE_COUNT)
CAVITY_NUMBER = anp.matmul(CAVITY_CREATE, CAVITY_ANNIHILATE)
CAVITY_ZERO = anp.array(((1,), (0,), (0,), (0,), (0,)))
CAVITY_ONE = anp.array(((0,), (1,), (0,), (0,), (0,)))
CAVITY_TWO = anp.array(((0,), (0,), (1,), (0,), (0,)))
CAVITY_THREE = anp.array(((0,), (0,), (0,), (1,), (0,)))
CAVITY_FOUR = anp.array(((0,), (0,), (0,), (0,), (1,)))
CAVITY_I = anp.eye(CAVITY_STATE_COUNT)

TRANSMON_STATE_COUNT = 3
TRANSMON_G = anp.array(((1,), (0,), (0,)))
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = anp.array(((0,), (1,), (0,)))
TRANSMON_E_DAGGER = conjugate_transpose(TRANSMON_E)
TRANSMON_F = anp.array(((0,), (0,), (1,)))
TRANSMON_F_DAGGER = conjugate_transpose(TRANSMON_F)
TRANSMON_I = anp.eye(TRANSMON_STATE_COUNT)

H_SYSTEM = (CHI_E * anp.kron(anp.matmul(TRANSMON_E, TRANSMON_E_DAGGER),
                             CAVITY_NUMBER)
            + CHI_F * anp.kron(anp.matmul(TRANSMON_F, TRANSMON_F_DAGGER),
                               CAVITY_NUMBER))
H_GE = anp.kron(anp.matmul(TRANSMON_G, TRANSMON_E_DAGGER), CAVITY_I)
H_GE_DAGGER = conjugate_transpose(H_GE)
H_EF = anp.kron(anp.matmul(TRANSMON_E, TRANSMON_F_DAGGER), CAVITY_I)
H_EF_DAGGER = conjugate_transpose(H_EF)
H_C = anp.kron(TRANSMON_I, CAVITY_ANNIHILATE)
H_C_DAGGER = conjugate_transpose(H_C)
hamiltonian = (lambda controls, time:
               H_SYSTEM
               + controls[0] * H_GE
               + anp.conjugate(controls[0]) * H_GE_DAGGER
               + controls[1] * H_EF
               + anp.conjugate(controls[1]) * H_EF_DAGGER
               + controls[2] * H_C
               + anp.conjugate(controls[2]) * H_C_DAGGER)
MAX_CONTROL_NORMS = anp.array((MAX_AMP_T, MAX_AMP_T, MAX_AMP_C,))

# Define the optimization.
ITERATION_COUNT = 5000
OPTIMIZER = Adam()
CONTROL_COUNT = 3
EVOLUTION_TIME = CONTROL_STEP_COUNT = 600

# Define the problem.
INITIAL_STATE_0 = anp.kron(TRANSMON_G, CAVITY_ZERO)
INITIAL_STATE_1 = anp.kron(TRANSMON_E, CAVITY_ZERO)
INITIAL_STATES = anp.stack((INITIAL_STATE_0, INITIAL_STATE_1,))
TARGET_STATE_0 = anp.kron(TRANSMON_G, CAVITY_ZERO)
TARGET_STATE_1 = anp.kron(TRANSMON_G, CAVITY_TWO)
TARGET_STATES = anp.stack((TARGET_STATE_0, TARGET_STATE_1,))
COSTS = (TargetStateInfidelity(TARGET_STATES),
         ControlVariation(CONTROL_COUNT,
                          CONTROL_STEP_COUNT,
                          MAX_CONTROL_NORMS,
                          cost_multiplier=0.5,
                          order=1,),
         ControlVariation(CONTROL_COUNT,
                          CONTROL_STEP_COUNT,
                          MAX_CONTROL_NORMS,
                          cost_multiplier=0.5,
                          order=2,),)

# Define the output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_FILE_NAME = "piccolo_exp7"
SAVE_PATH = os.path.join("./pulses/", SAVE_FILE_NAME,)
SAVE_FILE_PATH = generate_save_file_path(SAVE_FILE_NAME, SAVE_PATH,)


def main():
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_STEP_COUNT,
                                         COSTS, EVOLUTION_TIME,
                                         hamiltonian, INITIAL_STATES,
                                         ITERATION_COUNT,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_control_norms=MAX_CONTROL_NORMS,
                                         optimizer=OPTIMIZER,
                                         save_file_path=SAVE_FILE_PATH,
                                         save_iteration_step=SAVE_ITERATION_STEP,)


if __name__ == "__main__":
    main()
