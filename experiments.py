from argparse import ArgumentParser
import csv
from datetime import datetime
from trainTestReinforcementAlgorithm import *
import gym_snake.envs.snakeRewardFuncs as RewardFuncs
from stable_baselines3 import A2C, DQN, PPO

TRAIN_TIMESTEPS = 100000000
TEST_TIMESTEPS = 100000
BOARD_HEIGHT = 10
BOARD_WIDTH = 10
REPRESENT_BORDER = True
VISUALIZE_TESTING = False
VIS_FPS = 3000
CSV_FILENAME = "rl_data.csv"
SAVE_MODEL = True

def train_and_testRL(
    model_generator,
    reward_function,
    max_moves_no_fruit=0,    
    train_timesteps=TRAIN_TIMESTEPS,
    test_timesteps=TEST_TIMESTEPS,
    board_height=BOARD_HEIGHT,
    board_width=BOARD_WIDTH,
    represent_border=REPRESENT_BORDER,
    visualize_testing=VISUALIZE_TESTING,
    vis_fps=VIS_FPS,
    save_model=SAVE_MODEL,
    model_filename="",
    ):
    model = trainRL(model_generator=model_generator,
                    max_moves_no_fruit=max_moves_no_fruit,
                    train_timesteps=train_timesteps, 
                    board_height=board_height, 
                    board_width=board_width, 
                    visualization_fps=vis_fps, 
                    reward_function=reward_function,
                    represent_border=represent_border,)
    scores = testRL(model=model,
                    max_moves_no_fruit=max_moves_no_fruit,
                    test_timesteps=test_timesteps, 
                    board_height=board_height, 
                    board_width=board_width, 
                    visualize_testing=visualize_testing, 
                    visualization_fps=vis_fps, 
                    reward_function=reward_function,
                    represent_border=represent_border,)

    if save_model:
        saveRL(model, model_filename)

    return scores

def analyze_and_write_to_csv(strategy_label, strategy_description, scores):
    csv_file = open(CSV_FILENAME, "a+")
    csv_writer = csv.writer(csv_file)
    date_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S%z (%Z)]")
    analysis = analyzeRL(scores)
    csv_writer.writerow([date_time, strategy_label, strategy_description, analysis["completed_games"], analysis["high_score"], analysis["mean_score"], analysis["median_score"]])
    csv_file.close()
    print(strategy_label + "\n******\n\n")


def run_experiments(model_type, model_generator):
    # Kill and Punish .5x after 30 idle moves
    strategy_label = "("+model_type+"): "+"Punish half as much after 30 idle moves"
    strategy_description = "In this structure we punish the snake half as much as the reward for idle time with no fruit consumption"
    scores = train_and_testRL(model_generator, RewardFuncs.punish_half_for_move_ceiling, max_moves_no_fruit=30, model_filename=strategy_label)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)


def main():
    aparser = ArgumentParser("Experiments")
    aparser.add_argument("--commit_hash", type=str, default="NoneProvided")
    args = aparser.parse_args()


    csv_file = open(CSV_FILENAME, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["commit_hash", "Train Timesteps", "Test Timesteps", "Board Height", "Board Width", "Represent Border",])
    csv_writer.writerow([args.commit_hash, TRAIN_TIMESTEPS, TEST_TIMESTEPS, BOARD_HEIGHT, BOARD_WIDTH, REPRESENT_BORDER,])
    csv_writer.writerow([])
    csv_writer.writerow([])
    csv_writer.writerow(["Date/Time", "Strategy Label", "Strategy Description","Games Completed", "High Score","Mean Score", "Median Score",])
    csv_file.close()

    model_types = {
        # "A2C": lambda env: A2C("MlpPolicy", env, verbose=0),
        # "DQN": lambda env: DQN("MlpPolicy", env, verbose=0),
        "PPO": lambda env: PPO("MlpPolicy", env, verbose=0),
    }

    for model_type in model_types.keys():
        run_experiments(model_type, model_types[model_type])


if __name__ == "__main__":
    main()