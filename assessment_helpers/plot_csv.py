#**************************************************************************************
#plot_csv.py
#Author: Dave Carroll
#10/14/2021
#This script uses seaborn to plot GA data contianed in a .csv file
#*************************************************************************************
#Run this script using "python plot_csv.py" in the command line.
#Optionally specify the path of the .csv file to plot using the "--csv_filename"
#command line argument. The default is "../ga_data.csv" which is the .csv file
#automatically generated by "trainGeneticAlgorithm.py".
#Optionally specify the path of a .png file to save the plot in using the "--figure_filename"
#command line argument.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_filename", nargs='?', type=str, default="../ga_data.csv")
    parser.add_argument("--figure_filename", nargs='?', type=str, default="")
    args = parser.parse_args()
    
    sns.set_theme(color_codes=True)
    
    plotData = pd.read_csv(args.csv_filename)
    ax = sns.regplot(x="Generation", y="Average Score", data=plotData)

    plt.show()
    
    if (args.figure_filename):
        plt.savefig(args.figure_filename)
