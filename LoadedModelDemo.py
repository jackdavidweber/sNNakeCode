# %%
import argparse
import time
from typing import Callable
import gym
import gym_snake
from stable_baselines3 import A2C, PPO
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import gym_snake.envs.snakeRewardFuncs as RewardFuncs

# %% [markdown]
# ## Test
# Test the model to see how well it is performing. Also have the option to visualize the result

# %%
def testRL(
    model,
    test_timesteps=500, # Set amount of time for testing. One step is one action for the snake.
    env_name='snake-v0', # Set gym environment name.
    board_height=10, # Set game board height.
    board_width=10, # Set game board width.
    max_moves_no_fruit=30, # Set number of allowed moves without fruit consumption before ending the game. Any non-poitive number corresponds to no limit.
    visualize_testing=True, # Set to true in order to see game moves in pygame. Should be false if run on server.
    visualization_fps=20, # Set frames per second of testing visualization.
    reward_function=RewardFuncs.punish_tenth_for_move_ceiling, # Set reward function to be used in training. Reward functions are defined in snakeRewardFuncs.py
    represent_border=True, # Set a boolean flag for whether or not to represent the border in observation.
):
    # Setup
    env = gym.make(
        env_name, 
        board_height=board_height,
        board_width=board_width, 
        max_moves_no_fruit=max_moves_no_fruit,
        use_pygame=visualize_testing,
        fps=visualization_fps, 
        reward_func=reward_function,
        represent_border=represent_border,
    )
    obs = env.reset()
    
    # Run
    scores = []
    for i in range(test_timesteps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            scores.append(env.game.score)
            obs = env.reset()

    return scores

# %% [markdown]
# ## Analyze

# %%
def analyzeRL(
    scores,  # array of scores for each completed game
):
    s_arr = np.array(scores)

    analysis = {
        "completed_games": len(s_arr),
        "high_score": -1,
        "mean_score": -1,
        "median_score": -1,
    }
    print("Number of completed games: ", len(s_arr))

    if len(s_arr) > 0:
        analysis["high_score"]= np.max(s_arr)
        analysis["mean_score"]= np.average(s_arr)
        analysis["median_score"]= np.median(s_arr)
        print("High Score over all games: ", analysis["high_score"])
        print("Mean Score over all games: ", analysis["mean_score"])
        print("Median Score over all games: ", analysis["median_score"])

    return analysis

# %% [markdown]
# ## Load Model

# %%
# 5x5 trained model, 10M training steps
# model = A2C.load('a2c_5x5_model.zip')

# 10x10 trained model, 100M training steps
model = PPO.load('ppo_10x10_model.zip')

# %% [markdown]
# ## Run & Analyze Model

# %%
scores = testRL(model)
analyzeRL(scores)

# %%



