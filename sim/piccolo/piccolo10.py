"""
piccolo10.py - Experiment 10 for the piccolo system.
"""

from argparse import ArgumentParser
from copy import copy
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
META_NAME = "piccolo"
EXPERIMENT_NAME = "piccolo10"
MMQOC_PATH = os.environ["MULTIMODE_QOC_PATH"]
DATA_PATH = os.path.join(MMQOC_PATH, "out", EXPERIMENT_NAME)


# Specify computer specs.
CORE_COUNT = 8

# Define experimental constants.
BLOCKADE_LEVEL = 3
CHI_E = 2 * anp.pi * -5.644535742521878e-4 #GHz
KAPPA = 2 * anp.pi * -3.36e-6
OMEGA = 2 * anp.pi * 1.44e-4
MAX_AMP_C = anp.sqrt(2) * 2 * anp.pi * 1.5e-5
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
CAVITY_TWO = anp.copy(CAVITY_VACUUM)
CAVITY_TWO[2][0] = 1.

TRANSMON_STATE_COUNT = 2
TRANSMON_I = anp.eye(TRANSMON_STATE_COUNT)
TRANSMON_VACUUM = anp.zeros((TRANSMON_STATE_COUNT, 1))
TRANSMON_G = anp.copy(TRANSMON_VACUUM)
TRANSMON_G[0][0] = 1.
TRANSMON_G_DAGGER = conjugate_transpose(TRANSMON_G)
TRANSMON_E = anp.copy(TRANSMON_VACUUM)
TRANSMON_E[1][0] = 1.
TRANSMON_E_DAGGER = conjugate_transpose(TRANSMON_E)

H_SYSTEM = (
    2 * CHI_E * anp.kron(CAVITY_NUMBER - BLOCKADE_LEVEL * CAVITY_I,
                         anp.matmul(TRANSMON_E, TRANSMON_E_DAGGER))
    + OMEGA * anp.kron(CAVITY_I,
                       anp.matmul(TRANSMON_G, TRANSMON_E_DAGGER)
                       + anp.matmul(TRANSMON_E, TRANSMON_G_DAGGER))
    + KAPPA / 2 * anp.kron(anp.matmul(CAVITY_NUMBER, CAVITY_NUMBER - CAVITY_I), TRANSMON_I))
H_C_0 = anp.kron(CAVITY_ANNIHILATE, TRANSMON_I)
H_C_0_DAGGER = conjugate_transpose(H_C_0)

nhamiltonian = lambda controls, time: (
    H_SYSTEM
    + controls[0] * H_C_0
    + anp.conjugate(controls[0]) * H_C_0_DAGGER
)
CONTROL_COUNT = 1
COMPLEX_CONTROLS = True
MAX_CONTROL_NORMS = anp.array((MAX_AMP_C,))
MAX_CONTROL_BANDWIDTHS = anp.array((MAX_BND_C,))

# Define the problem.
EVOLUTION_TIME = int(3.5e4) #ns
CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = int(EVOLUTION_TIME)

INITIAL_STATE_0 = anp.kron(CAVITY_ZERO, TRANSMON_G)
INITIAL_STATE_1 = anp.kron(CAVITY_ONE, TRANSMON_G)
INITIAL_STATE_2 = anp.kron(CAVITY_TWO, TRANSMON_G)
INITIAL_STATES = anp.stack((
    INITIAL_STATE_0,
    INITIAL_STATE_1,
    INITIAL_STATE_2,
))
TARGET_STATE_0 = anp.kron(CAVITY_ONE, TRANSMON_G)
TARGET_STATE_1 = anp.kron(CAVITY_TWO, TRANSMON_G)
TARGET_STATE_2 = anp.kron(CAVITY_ZERO, TRANSMON_G)
TARGET_STATES = anp.stack((
    TARGET_STATE_0,
    TARGET_STATE_1,
    TARGET_STATE_2,
))
def gauss(x):
    b = anp.mean(x)
    c = anp.std(x)
    return anp.exp(-((x - b) ** 2) / (c ** 2))
CONTROL_NORM_WEIGHTS = 1 - gauss(anp.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT))
CONTROL_NORM_WEIGHTS = anp.repeat(CONTROL_NORM_WEIGHTS[:, anp.newaxis], CONTROL_COUNT, axis=1)
# Don't penalize controls in the middle.
# CONTROL_OFFSET = int(CONTROL_COUNT / 5)
# CONTROL_NORM_WEIGHTS[CONTROL_OFFSET:-CONTROL_OFFSET] = 0

FIDELITY_MULTIPLIER = 1.
CONTROL_VAR_01_MULTIPLIER = 1.
CONTROL_VAR_02_MULTIPLIER = 1.
CONTROL_NORM_MULTIPLIER = 0.5
CONTROL_BANDWIDTH_MULTIPLIER = 1.

COSTS = [
    TargetStateInfidelity(TARGET_STATES,
                          cost_multiplier=FIDELITY_MULTIPLIER),
     # ControlVariation(CONTROL_COUNT,
     #                  CONTROL_EVAL_COUNT,
     #                  cost_multiplier=CONTROL_VAR_01_MULTIPLIER,
     #                  max_control_norms=MAX_CONTROL_NORMS,
     #                  order=1),
     # ControlVariation(CONTROL_COUNT,
     #                  CONTROL_EVAL_COUNT,
     #                  cost_multiplier=CONTROL_VAR_02_MULTIPLIER,
     #                  max_control_norms=MAX_CONTROL_NORMS,
     #                  order=2),
     # ControlNorm(CONTROL_COUNT, CONTROL_EVAL_COUNT,
     #             control_weights=CONTROL_NORM_WEIGHTS,
     #             cost_multiplier=CONTROL_NORM_MULTIPLIER,
     #             max_control_norms=MAX_CONTROL_NORMS,),
     # ControlBandwidthMax(CONTROL_COUNT, CONTROL_EVAL_COUNT, EVOLUTION_TIME,
     #                     MAX_CONTROL_BANDWIDTHS,
     #                     cost_multiplier=CONTROL_BANDWIDTH_MULTIPLIER),
]

def impose_control_conditions(controls):
    # Impose 0 at the boundaries.
    controls[0, :]= 0
    controls[-1, :] = 0
    return controls

# Define the optimization.
LEARNING_RATE = 5e-2
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
ITERATION_COUNT = int(1e3)
GRAB_CONTROLS = False
if GRAB_CONTROLS:
    controls_path = os.path.join(DATA_PATH, "00003_piccolo10.h5")
    controls_lock_path = "{}.lock".format(controls_path)
    try:
        with FileLock(controls_lock_path):
            with h5py.File(controls_path, "r") as file_:
                index = anp.argmin(file_["error"])
                controls = file_["controls"][index]
    except Timeout:
        print("Timeout encountered")
        exit(0)
    INITIAL_CONTROLS = controls
else:
    INITIAL_CONTROLS = None
    
# Define the output.
LOG_ITERATION_STEP = 1
SAVE_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES_GRAPE = False
SAVE_INTERMEDIATE_STATES_EVOL = True
SAVE_EVOL = False

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
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_GRAPE,
    "save_iteration_step": SAVE_ITERATION_STEP,
}

EVOL_CONFIG = {
    "controls": INITIAL_CONTROLS,
    "costs": COSTS,
    "evolution_time": EVOLUTION_TIME,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "system_eval_count": SYSTEM_EVAL_COUNT,
    "save_intermediate_states": SAVE_INTERMEDIATE_STATES_EVOL,
}

def run_grape():
    save_file_path = generate_save_file_path(EXPERIMENT_NAME, DATA_PATH)
    config = copy(GRAPE_CONFIG)
    config.update({
        "save_file_path": save_file_path,
    })
    result = grape_schroedinger_discrete(**config)


def run_evolve():
    config = copy(EVOL_CONFIG)
    if SAVE_EVOL:
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, DATA_PATH)
        config.update({
            "save_file_path": save_file_path,
        })
    result = evolve_schroedinger_discrete(**config)
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
    os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(CORE_COUNT)
    os.environ["MKL_NUM_THREADS"] = "{}".format(CORE_COUNT)
    main()
