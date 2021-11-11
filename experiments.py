import csv
from datetime import datetime
from trainTestReinforcementAlgorithm import *

TRAIN_TIMESTEPS = 100 # 10000000
TEST_TIMESTEPS = 10 # 10000
VISUALIZE_TESTING = False
CSV_FILENAME = "rl_data.csv"


def analyze_and_write_to_csv(strategy_label, strategy_description, scores):
    csv_file = open(CSV_FILENAME, "a+")
    csv_writer = csv.writer(csv_file)
    date_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S%z (%Z)]")
    analysis = analyzeRL(scores)
    csv_writer.writerow([date_time, strategy_label, strategy_description, analysis["completed_games"], analysis["high_score"], analysis["mean_score"], analysis["median_score"]])
    csv_file.close()
    print(strategy_label + "\n******\n\n")


def main():
    csv_file = open(CSV_FILENAME, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Date/Time","Strategy Label", "Strategy Description","Games Completed", "High Score","Mean Score", "Median Score",])
    csv_file.close()  

    # Basic Rewards Structure
    strategy_label = "Basic Rewards Structure"
    strategy_description = "Here we just do the basic reward structure of + for fruit and - for wall. We do not kill the snake after a set number of moves."
    model = trainRL(train_timesteps=TRAIN_TIMESTEPS)
    scores = testRL(model=model, test_timesteps=TEST_TIMESTEPS, visualize_testing=VISUALIZE_TESTING)
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)

    # Other Rewards Structure  TODO: FILL IN THE TODO Lines
    strategy_label = ""  # TODO
    strategy_description = ""  # TODO
    model = trainRL(train_timesteps=TRAIN_TIMESTEPS)  # TODO
    scores = testRL(model=model, test_timesteps=TEST_TIMESTEPS, visualize_testing=VISUALIZE_TESTING)  # TODO: fill in
    analyze_and_write_to_csv(strategy_label, strategy_description, scores)



if __name__ == "__main__":
    main()