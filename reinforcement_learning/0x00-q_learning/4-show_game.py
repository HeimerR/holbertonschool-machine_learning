#!/usr/bin/env python3
""" play """
import numpy as np


def play(env, Q, max_steps=100):
    """ has the trained agent play an episode

        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - max_steps is the maximum number of steps in the episode

        Each state of the board should be displayed via the console
        You should always exploit the Q-table

        Returns: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        # Choose action with highest Q-value for current state
        action = np.argmax(Q[state, :])
        # Take new action
        state, reward, done, info = env.step(action)
        # Show current state of environment on screen
        env.render()
        if done:
            break
    return reward
