#!/usr/bin/env python3
import tensorflow
print(tensorflow.__version__)
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from tensorflow.keras import layers
import tensorflow.keras as K
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from PIL import Image
import numpy as np

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

"""
def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))
"""
def create_q_model(num_actions, window):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(window, 84, 84))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", data_format="channels_first")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", data_format="channels_first")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", data_format="channels_first")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)

def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

if __name__ == '__main__':
    file_name = "policy"
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    print(num_actions)
    window = 4
    model = create_q_model(num_actions, window)
    print(model.summary())
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                              nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    callbacks = build_callbacks(file_name)

    dqn.fit(env,
            nb_steps=17500,
            log_interval=10000,
            visualize=False,
            verbose=2,
            callbacks=callbacks)

    # After training is done, we save the final weights.
    dqn.save_weights('policy.h5', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
