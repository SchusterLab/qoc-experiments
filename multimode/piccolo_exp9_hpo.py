"""
piccolo_exp9_hpo.py - Experiment 9 for the piccolo system with hpo.
"""

# Set random seeds for reasonable reproducibility.
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import os
os.environ["HYPEROPT_FMIN_SEED"] = "23"

import argparse
from copy import copy
import json
import sys
import time

from filelock import FileLock, Timeout
from hyperopt import hp
import h5py
import numpy as np
from qoc import grape_schroedinger_discrete
from qoc.standard import (Adam, CustomJSONEncoder,
                          TargetStateInfidelity,
                          ControlVariation,
                          ControlNorm,)
import ray
import ray.tune
from ray.tune import Trainable, register_trainable
from ray.tune.suggest.hyperopt import HyperOptSearch

from piccolo_exp9 import (GRAPE_CONFIG,
                          TARGET_STATES,
                          FIDELITY_MULTIPLIER,
                          CONTROL_COUNT,
                          CONTROL_VAR_01_MULTIPLIER,
                          MAX_CONTROL_NORMS,
                          CONTROL_VAR_02_MULTIPLIER,
                          CONTROL_NORM_WEIGHTS,
                          CONTROL_NORM_MULTIPLIER,
                          MAX_CONTROL_NORMS,
                          gauss)

# Machine parameters.
CORE_COUNT = 8
MAX_CONCURRENT = 2
RESOURCES_PER_TRIAL = {
    "cpu": 4,
}

# Data paths.
META_NAME = "piccolo_exp9_hpo"
DATA_PATH = os.path.join(os.environ["MULTIMODE_QOC_PATH"], "out", META_NAME)
LOG_FILE_NAME = "{}.log".format(META_NAME)
LOG_FILE_PATH = os.path.join(DATA_PATH, LOG_FILE_NAME)
LOG_FILE_LOCK_NAME = "{}.log.lock".format(META_NAME)
LOG_FILE_LOCK_PATH = os.path.join(DATA_PATH, LOG_FILE_LOCK_NAME)

# Define the hyperparameter search space.
LR_LB = 1e-8
LR_UB = 1
TIME_LB = int(2e4) #ns
TIME_UB = int(5e4) #ns
HP_SEARCH_SPACE = {
    "lr": hp.loguniform("lr", np.log(LR_LB), np.log(LR_UB)),
    "time": hp.uniform("time", TIME_LB, TIME_UB),
}

# Define optimization.
HPO_MAX_ITERATIONS = int(1e6)
QOC_MAX_ITERATIONS = int(3)
GRAPE_CONFIG.update({
    "initial_controls": None,
    "iteration_count": QOC_MAX_ITERATIONS,
    "save_intermediate_states": False,
    "save_iteration_step": 0,
})

# Define the hp search algorithm.
HP_ALGO = HyperOptSearch(HP_SEARCH_SPACE,
                         max_concurrent=MAX_CONCURRENT,
                         metric="error",
                         mode="min")

# Configure ray.
# Give 100MB to each the object store and redis shard.
OBJECT_STORE_MEMORY = int(1e8)
REDIS_MAX_MEMORY = int(1e8)
HP_RUN_CONFIG = {
    "num_samples": HPO_MAX_ITERATIONS,
    "name": META_NAME,
    "search_alg": HP_ALGO,
    "verbose": 1,
    "local_dir": DATA_PATH,
    "resources_per_trial": RESOURCES_PER_TRIAL,
    "resume": False,
}

class QOCTrainable(ray.tune.Trainable):
    def _setup(self, config):
        self.qoc_config = copy(GRAPE_CONFIG)
        self.log_config = {}
        learning_rate = config["lr"]
        evolution_time = config["time"]
        optimizer = Adam(learning_rate=learning_rate)
        system_eval_count = control_eval_count = int(evolution_time)
        costs = get_costs(control_eval_count, evolution_time)
        self.qoc_config.update({
            "control_eval_count": control_eval_count,
            "costs": costs,
            "evolution_time": evolution_time,
            "optimizer": optimizer,
            "system_eval_count": system_eval_count,
        })
        self.log_config.update({
            "lr": learning_rate,
            "time": evolution_time,
        })


    def _train(self):
        # Get the result of optimization.
        result = grape_schroedinger_discrete(**self.qoc_config)
        error = result.best_error

        # Log the result.
        self.log_config.update({
            "error": error,
        })
        log_str = "{}\n".format(json.dumps(self.log_config, cls=CustomJSONEncoder))
        try:
            with FileLock(LOG_FILE_LOCK_PATH):
                with open(LOG_FILE_PATH, "a+") as log_file:
                    log_file.write(log_str)
        except Timeout:
            print("log file timeout:\n{}"
                  "".format(self.log_config))

        # Report the result.
        result_dict = {
            "error": error,
            "done": True,
        }
        
        return result_dict


def main():
    # Start ray and run HPO.
    ray.init(num_cpus=CORE_COUNT, object_store_memory=OBJECT_STORE_MEMORY,
             redis_max_memory=REDIS_MAX_MEMORY)
    ray.tune.run(QOCTrainable, **HP_RUN_CONFIG)


def get_costs(control_eval_count, evolution_time):
    control_norm_weights = 1 - gauss(np.linspace(0, evolution_time, control_eval_count))
    control_norm_weights = np.repeat(control_norm_weights[:, np.newaxis], CONTROL_COUNT, axis=1)
    # Don't penalize controls in the middle.
    control_offset = int(control_eval_count / 5)
    control_norm_weights[control_offset:-control_offset] = 0
    return [TargetStateInfidelity(TARGET_STATES,
                                  cost_multiplier=FIDELITY_MULTIPLIER),
            ControlVariation(CONTROL_COUNT,
                             control_eval_count,
                             cost_multiplier=CONTROL_VAR_01_MULTIPLIER,
                             max_control_norms=MAX_CONTROL_NORMS,
                             order=1),
            ControlVariation(CONTROL_COUNT,
                             control_eval_count,
                             cost_multiplier=CONTROL_VAR_02_MULTIPLIER,
                             max_control_norms=MAX_CONTROL_NORMS,
                             order=2),
            ControlNorm(CONTROL_COUNT,
                        control_eval_count,
                        control_weights=control_norm_weights,
                        cost_multiplier=CONTROL_NORM_MULTIPLIER,
                        max_control_norms=MAX_CONTROL_NORMS,),
        ]


if __name__ == "__main__":
    main()
