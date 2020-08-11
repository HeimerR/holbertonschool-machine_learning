#!/usr/bin/env python3
""" play """
import numpy as np
import gym
import random
import time
from gym.envs.toy_text import frozen_lake
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


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
    for step in range(max_steps):
        env.render()
        time.sleep(0.3)
	# Show current state of environment on screen
        action = np.argmax(Q[state,:])
        new_state, reward, done, info = env.step(action)
        # Choose action with highest Q-value for current state
	# Take new action

        if done:
            # print(reward)
            env.render()
            break
            """
            if reward == 1:
		# Agent reached the goal and won episode
            else:
		# Agent stepped in a hole and lost episode

	    # Set new state
            """
    env.close()
    return reward
