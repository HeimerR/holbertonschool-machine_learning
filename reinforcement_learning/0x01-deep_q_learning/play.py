#!/usr/bin/env python3
import gym
env = gym.make("Breakout-v0")
env.reset()
is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  # Render
  env.render()
