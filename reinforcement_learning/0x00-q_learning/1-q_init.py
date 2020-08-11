#!/usr/bin/env python3
""" Initialize Q-table """
import numpy as np
import gym
import random
import time
from gym.envs.toy_text import frozen_lake


def q_init(env):
    """ initializes the Q-table
        - env is the FrozenLakeEnv instance
        Returns: the Q-table as a numpy.ndarray of zeros
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
