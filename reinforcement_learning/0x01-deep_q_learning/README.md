Breakout - Deep Q-learning

a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:

	the script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
	the script should save the final policy network as policy.h5
python script play.py that can display a game played by the agent trained by train.py:

	the script should load the policy network saved in policy.h5
	the agent should use the GreedyQPolicy
