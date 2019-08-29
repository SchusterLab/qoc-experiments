"""
piccolo_exp1_hpo.py - Experiment 1 for the piccolo system
with hyperparameter optimization.
"""

# Set random seeds for reasonable reproducability.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.compat.v1.set_random_seed(2)
import os
os.environ["HYPEROPT_FMIN_SEED"] = "23"

from copy import deepcopy
import json
import time

import autograd.numpy as anp
from hyperopt import hp
from qoc import grape_schroedinger_discrete
from qoc.standard import (conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          TargetInfidelity,
                          ParamVariation,
                          LBFGSB, Adam,
                          CustomJSONEncoder,)
import ray
import ray.tune
from ray.tune.suggest.hyperopt import HyperOptSearch

# Specify computer specs.
CORE_COUNT = 8

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
hamiltonian = (lambda params, t:
               H_SYSTEM
               + params[0] * H_GE
               + anp.conjugate(params[0]) * H_GE_DAGGER
               + params[1] * H_EF
               + anp.conjugate(params[1]) * H_EF_DAGGER
               + params[2] * H_C
               + anp.conjugate(params[2]) * H_C_DAGGER)
MAX_PARAM_NORMS = anp.array((MAX_AMP_T, MAX_AMP_T, MAX_AMP_C,))

# Define the optimization.
ITERATION_COUNT = 1000
PARAM_COUNT = 3
PULSE_TIME = 250
PULSE_STEP_COUNT = PULSE_TIME

# Define the problem.
INITIAL_STATE_0 = anp.kron(TRANSMON_G, CAVITY_ZERO)
INITIAL_STATES = anp.stack((INITIAL_STATE_0,))
TARGET_STATE_0 = anp.kron(TRANSMON_G, CAVITY_ONE)
TARGET_STATES = anp.stack((TARGET_STATE_0,))
COSTS = (TargetInfidelity(TARGET_STATES),
         ParamVariation(MAX_PARAM_NORMS,
                        PARAM_COUNT,
                        PULSE_STEP_COUNT,))

# Define the output.
LOG_ITERATION_STEP = 0

GRAPE_CONFIG = {
    "costs": COSTS,
    "hamiltonian": hamiltonian,
    "initial_states": INITIAL_STATES,
    "iteration_count": ITERATION_COUNT,
    "param_count": PARAM_COUNT,
    "pulse_step_count": PULSE_STEP_COUNT,
    "pulse_time": PULSE_TIME,
    "log_iteration_step": LOG_ITERATION_STEP,
    "max_param_norms": MAX_PARAM_NORMS,
}

# Define HPO constants and search space.
LR_LB = 1e-5
LR_UB = 1
HPO_MAX_ITERATIONS = int(1e6)

# Define Ray parameters.
# Allocate 0.5gb to object store and redis shard respectively.
OBJECT_STORE_MEMORY = int(5e8)
REDIS_MAX_MEMORY = int(5e8)
EFFECTIVE_CORE_COUNT = CORE_COUNT - 1

class ProcessState(object):
    """An object to encapsulate the HPO of a grape problem.
    Fields:
    grape_config :: dict - the parameters to pass to grape
    file_name :: str - the experiment name
    data_path :: str - the path to store output
    log_path :: str - the path to the log file for hpo output
    """

    def __init__(self, grape_config):
        """See class field definitions for corresponding arguments.
        """
        super()
        self.grape_config = grape_config
        # inherit file name and data path from grape config
        self.file_name = grape_config["file_name"]
        self.data_path = grape_config["data_path"]
        log_file = "{}.out".format(self.file_name)
        self.log_path = os.path.join(self.data_path, log_file)



def main():
    result = grape_schroedinger_discrete(COSTS, hamiltonian, INITIAL_STATES,
                                         ITERATION_COUNT, PARAM_COUNT,
                                         PULSE_STEP_COUNT,
                                         PULSE_TIME,
                                         log_iteration_step=LOG_ITERATION_STEP,
                                         max_param_norms=MAX_PARAM_NORMS,)


if __name__ == "__main__":
    main()
