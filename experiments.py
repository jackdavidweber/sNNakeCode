import csv
from datetime import datetime
from trainTestReinforcementAlgorithm import *
import gym_snake.envs.snakeRewardFuncs as RewardFuncs
from stable_baselines3 import DQN, A2C, PPO

TRAIN_TIMESTEPS = 1000000
TEST_TIMESTEPS = 100000
BOARD_HEIGHT = 10
BOARD_WIDTH = 10
VISUALIZE_TESTING = False
VIS_FPS = 3000
CSV_FILENAME = "rl_data.csv"


def analyze_and_write_to_csv(strategy_label, strategy_description, scores):
    csv_file = open(CSV_FILENAME, "a+")
    csv_writer = csv.writer(csv_file)
    date_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S%z (%Z)]")
    analysis = analyzeRL(scores)
    csv_writer.writerow([date_time, strategy_label, strategy_description, analysis["completed_games"], analysis["high_score"], analysis["mean_score"], analysis["median_score"]])
    csv_file.close()
    print(strategy_label + "\n******\n\n")


def run_experiments(model_type, model_generator):
    # Basic Rewards Structure
    strategy_label = "("+model_type+"): "+"Basic Rewards Structure"
    strategy_description = "Here we just do the basic reward structure of + for fruit and - for wall. We do not kill the snake after a set number of moves."
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Distance Reward Structure
    strategy_label = "("+model_type+"): "+"Reward Closer to Fruit"
    strategy_description = "Very similar to the basic reward structure, but we reward when the snake moves a step closer to the fruit"
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.reward_closer_to_fruit)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.reward_closer_to_fruit)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Kill after 10 idle moves
    strategy_label = "("+model_type+"): "+"Kill after 10 idle moves"
    strategy_description = "Here we just do the basic reward structure of + for fruit and - for wall. We kill the snake after a set number of idle moves."
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=10,
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    max_moves_no_fruit=10,
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Kill after 30 idle moves
    strategy_label = "("+model_type+"): "+"Kill after 30 idle moves"
    strategy_description = "Here we just do the basic reward structure of + for fruit and - for wall. We kill the snake after a set number of idle moves."
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=30,
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH, 
                    max_moves_no_fruit=30,
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Kill and Punish after 10 idle moves
    strategy_label = "("+model_type+"): "+"Punish after 10 idle moves"
    strategy_description = "In this structure we punish the snake for idle time with no fruit consumption"
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=10,
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func_with_move_ceiling)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=10,
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func_with_move_ceiling)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Kill and Punish after 30 idle moves
    strategy_label = "("+model_type+"): "+"Punish after 30 idle moves"
    strategy_description = "In this structure we punish the snake for idle time with no fruit consumption"
    model = trainRL(model_generator=model_generator,
                    train_timesteps=TRAIN_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=30,
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func_with_move_ceiling)
    scores = testRL(model=model, 
                    test_timesteps=TEST_TIMESTEPS, 
                    board_height=BOARD_HEIGHT, 
                    board_width=BOARD_WIDTH,
                    max_moves_no_fruit=30,
                    visualize_testing=VISUALIZE_TESTING, 
                    visualization_fps=VIS_FPS, 
                    reward_function=RewardFuncs.basic_reward_func_with_move_ceiling)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)



def main():
    csv_file = open(CSV_FILENAME, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Date/Time","Strategy Label", "Strategy Description","Games Completed", "High Score","Mean Score", "Median Score",])
    csv_file.close()

    model_types = {
        "A2C": lambda env: A2C("MlpPolicy", env, verbose=0),
        "DQN": lambda env: DQN("MlpPolicy", env, verbose=0),
        "PPO": lambda env: PPO("MlpPolicy", env, verbose=0),
    }

    for model_type in model_types.keys():
        run_experiments(model_type, model_types[model_type])


if __name__ == "__main__":
    main()