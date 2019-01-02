from env.env import *
import logging
import numpy as np
from env.slot_filling_2.env_slot_filling import EnvSlotFilling


def create_mock_env(size_constraints):
    mock_env = EnvSlotFilling(None,
                              reward_by_turn=None,
                              max_size=None,
                              size_constraints=size_constraints)

    return mock_env


def pretty_print_trajectory(trajectory):
    print("------------------------")
    user_acts = trajectory[-1].s_.user_acts
    machine_acts = trajectory[-1].s_.machine_acts
    for i in range(0, len(user_acts)):
        print ('system says : {}'.format(machine_acts[i]))
        print ('user says : {}'.format(user_acts[i]))
    if len(machine_acts) > len(user_acts):
        print ('system says : {}'.format(machine_acts[-1]))


def generate_proba_hangup(mean, std, k=0):
    xx = np.random.normal(mean, std)
    if xx < 0.:
        if k > 5:
            return 0.
        else:
            return generate_proba_hangup(mean / 2., std / 2., k=k + 1)
    if xx > 1.:
        if k > 5:
            return 1.0
        else:
            return generate_proba_hangup(mean + (1. - mean / 2.), std / 2., k=k + 1)
    return xx
