#!/usr/bin/env python3
""" Q-learning """
import numpy as np
import gym
import random
import time
from gym.envs.toy_text import frozen_lake
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """  performs Q-learning
        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should decay to
        - epsilon_decay is the decay rate for updating epsilon between
            episodes

        When the agent falls in a hole, the reward should be updated to be -1
        Returns: Q, total_rewards
            Q is the updated Q-table
            total_rewards is a list containing the rewards per episode
    """
    rewards_all_episodes = []
    # Q-learning algorithm
    for episode in range(episodes):
        # initialize new episode params
        state = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps):
            # Exploration-exploitation trade-off
            action = epsilon_greedy(Q, state, epsilon)
            # Take new action
            new_state, reward, done, info = env.step(action)
            # Update Q-table
            Q[state, action] = Q[state, action] * (1 - alpha) + \
                    alpha * (reward + gamma * np.max(Q[new_state, :]))
            # Set new state
            state = new_state
            rewards_current_episode += reward
            # Add new reward

        # Exploration rate decay
        exploration_rate = min_epsilon + \
                    (1 - min_epsilon) * np.exp(-epsilon_decay*episode)
        # Add current episode reward to total rewards list
        rewards_all_episodes.append(rewards_current_episode)
    return Q, rewards_all_episodes
