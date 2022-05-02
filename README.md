# sNNake: Reinforcement Learning Strategies in Screen Snake
#### *By Jack Weber, Dave Carroll, and David D'Attile*

| Untrained sNNake | Trained sNNake |
|:----------------:|:--------------:|
| ![Untrained sNNake](https://github.com/davidd-55/sNNake/blob/main/media/10x10_dumb_snake_compressed.gif?raw=true) | ![Trained sNNake](https://github.com/davidd-55/sNNake/blob/main/media/10x10_smart_snake_compressed.gif?raw=true) |

###### *[- Link to presentation and demo](https://youtu.be/6J6peCubHf0)*
###### *[- Link to slide deck](https://docs.google.com/presentation/d/1aH9gAsUfNUAUEYVZmVjEOOKfWeldbtVQxH1p8VCeUGg/edit?usp=sharing)*
###### *[- Link to raw experiment data](https://docs.google.com/spreadsheets/d/1yNhVSQuhbs6fgsL-IMh18bhgLSnmTLftfySUerWKHOE/edit?usp=sharing)*
###### *[- Link to repository](https://github.com/jackdavidweber/sNNakeCode)*

## Contents:
- [1. Abstract](#1-abstract)
- [2. Introduction](#2-introduction)
- [3. Related Works](#3-related-works)
- [4. Methods](#4-methods)
- [5. Discussion](#5-discussion)
- [6. Ethics](#6-ethics)
- [7. Reflection](#7-reflection)
- [8. Appendix](#8-appendix)

## 1. Abstract

This project explores the application of reinforcement learning (RL) algorithms to the well-known game screen snake. After a brief introduction to RL concepts and related studies, it analyzes the performance of snake agents trained with different combinations of reinforcement algorithms and reward functions. Finally, this project provides a discussion on the ethical questions associated with the application of RL techniques across a wide range of fields and a reflection on the work conducted.

## 2. Introduction
#### 2a. Background
Even if you are not interested in computer science, you have almost definitely interacted with reinforcement algorithms in your day-to-day life. These algorithms suggest which Netflix shows we will like. They help advertisers determine the optimal products to present us with. They are better than humans at [Chess](https://www.ibm.com/ibm/history/ibm100/us/en/icons/deepblue/), [Go](https://www.youtube.com/watch?v=WXuK6gekU1Y&ab_channel=DeepMind), and [driving](https://www.scientificamerican.com/article/are-autonomous-cars-really-safer-than-human-drivers/).

Given the importance of these algorithms in our lives, sNNake is a project aimed at better understanding how reinforcement algorithms work. We do this by focusing on the goal of teaching an agent to play [Screen-Snake (SS)](https://en.wikipedia.org/wiki/Snake_(video_game_genre)). SS is a simple game where the snake tries to grow in length by eating more _fruits_. The snake dies when it collides with the boundary wall or part of its own body.

So, how do we teach an agent to play SS? Well, one way to do it would be to try to come up with a set of rules that we feed directly into the agent. Rules like “if collision is imminent, turn left” or “if moving away from fruit, turn around.” This would be quite difficult and take a lot of time.

Another option is to get the agent to learn by trial and error. We let the agent play randomly and whenever it does something positive (like get a fruit) we reward it. Whenever the agent does something negative (like collide with a wall), we punish it. Over time, by reinforcing positive actions and disincentivizing negative actions, the agent will start to figure out the best strategies to get more positive rewards. The agent will learn how to play SS without us ever having to explicitly teach it.

This is the essence of RL algorithms. The algorithm requires us to provide rewards for actions. It then tries many strategies to complete a given task. The strategies that are rewarded are given more weight and the algorithm becomes better at the assigned task.

#### 2b. Details
Rather than implementing SS from scratch, we based our implementation on [AI for Snake Game](https://github.com/craighaber/AI-for-Snake-Game) by Craig Haber. Built on top of PyGame, this repo provided us with a basic implementation and visualization of the game (see image below). It also came with files to train and test a snake agent built with a genetic algorithm. While initially we experimented with it, we eventually concluded that the genetic algorithms were beyond the scope of our project (more details in methods section). 

<p align="center">
  <img alt="AI for Snake Game Board" width="320" height="400" src="https://github.com/davidd-55/sNNake/blob/main/media/10x10_trained_representation_compressed.png?raw=true">
</p>

Instead, we used [OpenAI Gym](https://gym.openai.com/) for the RL infrastructure. Figuring out how to make Haber's repo work with Open AI Gym was initially challenging, but we ended up with a system that enabled easy and rapid experimentation. Our final repo is highly parameterized, allowing easy changes of board size, RL algorithm, reward function, and board representation.

#### 2c. Assessment
On a 5x5 board, trained agents were able to get a high score of 23 and a median score of 18. This is a pretty great result. A 5x5 board has 25 board spaces, so a high score of 23 is a mere 2 points away from beating the game. In the methods section we will go into more depth on how we implemented the system capable of achieving such a result. In the discussion section we will go through the many experiments that we ran to go from a snake that continuously ran into walls to a snake that has relative mastery of the game.

## 3. Related Works
For related works, our team opted to investigate both academic literature and implementations related to our project. 

#### 3a. Academic Articles

In the first article we reviewed, [A Hybrid Algorithm Using a Genetic Algorithm and Multiagent Reinforcement Learning Heuristic to Solve the Traveling Salesman Problem](https://link.springer.com/article/10.1007/s00521-017-2880-4) the authors explain how they use multi-agent RL and genetic algorithms in order to solve the traveling salesman problem. This is helpful for our project since we would like to experiment with evolving RL algorithms with genetic algorithms. We are also interested in learning about the interactions between multiple agents, or snakes in our case.

In [Autonomous Agents in Snake Game via Deep Reinforcement Learning](https://www.researchgate.net/publication/327638529_Autonomous_Agents_in_Snake_Game_via_Deep_Reinforcement_Learning), the second paper we reviewed, the authors leverage a Deep Q-Learning Network (DQN) to teach an agent how to play SS. The researchers’ implementation uses a series of four pre-processed screenshots of the game board as inputs, 3 convolutional layers & 1 fully-connected layer (all using the ReLU activation function), and 4 outputs corresponding to the snake’s movements. The authors leveraged a novel training concept they referred to as a “training gap” - this allowed the snakes to only focus on non game-ending movement immediately after successfully eating a piece of fruit, since the direction of the respawned fruit could often lead snakes to collide with their tail.

#### 3b. Existing Implementations

We chose to explore open-source implementations with the goal of selecting a project to build sNNake upon. This approach allowed us to focus less on building a SS game from scratch and more on the implementation of NNs teaching agents how to play the game. 

First, we explored the [SnakeAI](https://github.com/greerviau/SnakeAI) tutorial which walks through how to train a neural network to play SS using Deep Learning and also utilizes a genetic algorithm to combine and mutate successful neural networks. This project is coded in Processing, a language our team is not experienced with for development. This seems like an unlikely project to build on but could be useful for cross referencing learning algorithms and techniques.

Second, we investigated [How to Teach AI to Play Games: Deep Reinforcement Learning](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a). This tutorial walks through how to train a neural network to play Snake using a Deep-Q RL Network (DQN) algorithm coded in Python. The tutorial displays very promising results accomplished with little training time. However, running our own version, we were not able to replicate the results to the same standard and we experienced problems with the game display. Interestingly, rather than an array or image representation of the board used as input, this implementation used an array containing 11 boolean values coorrespondiong to immediates dangers, current snake direction, and fruit location relative to the snake.

Finally, [AI for Snake Game](https://github.com/craighaber/AI-for-Snake-Game) walks through how to train a neural network to play SS using Deep Learning while also utilizing a genetic algorithm to combine and mutate successful neural networks. Importantly, this Python code base is clean and well documented. This seems like an ideal project to base our project on.

Among all projects surveyed, we chose to base sNNake off of AI for Snake Game for the following reasons:
* Code base is clearly commented and cleanly written
* Code base written in Python 3
* Implementation contained easily modifiable game board/logic rules
* Implementation contained a genetic NN component for training that we can study
* Implementation contained a framework for easily expanding existing NN capabilities such as integrating the OpenAI Gym framework for RL networks

We believe that the combination of reviewing academic research and experimenting with pre existing applications have provided our team ample background to pursue the implementation of sNNake.

## 4. Methods

#### 4a. Initial Software
While the original AI for Snake Game repo was incredibly successful in training agents to play SS, unfortunately the learning paradigm was not ideal for our project for a few reasons:

* The entire repo was built from scratch, meaning it would have been highly impractical for us to experiment with different neural networks, or really to dynamically alter anything about the learning paradigm.
* As we dug deeper into the repo, we realized that the learning paradigm that had been implemented by scratch was highly niche and non-standard. Specifically, we realized that the NN component was not programmed to be updated via backpropagation, rather the learning paradigm initializes many networks each corresponding to a game, uses forward propagation on each to get actions for each agent, records performance metrics for each network, then creates a new “generation” of networks using a genetic algorithm which operates on those that have just been tested. Not only could we not utilize this learning paradigm to implement an RL algorithm, we couldn’t even apply our knowledge of neural networks to parse the learning paradigm since it relied so heavily on genetic algorithms, a topic we have not formally seen in any class.
* We were interested in leveraging RL to train snake agents and given our relative inexpertise in RL and the difficulty of gaining expertise, it would have been infeasible to implement a RL algorithm from scratch in the way that AI for Snake Game implemented its learning paradigm.

#### 4b. Final Software

As a result of the above considerations, we decided to move in a new direction and use the existing SS game infrastructure from AI for Snake Game along with OpenAI Gym, a python toolkit which supplies RL frameworks for general tasks. We acknowledge that the intimate understanding of reinforcement algorithms that would be necessary to implement this from scratch is far beyond us at the moment. Despite this, utilizing the OpenAI Gym toolkit will allow us to experiment with RL algorithms at a high level. So, we isolated the SS game infrastructure and ruleset, then applied different OpenAI Gym RL algorithms to train snake agents and study their behavior. 

At a high level, an RL implementation of SS requires consistent input and output at each "step" of the game (i.e. each move the snake takes) and teach an agent desired behaviors. As input, at each step we provided a 2-dimensional array representation of the current state of the board where different integers coorespond to board slot states such as empty, snake body, snake head, or fruit. After the network receives this board representation as input, it outputs a single array of four values corresponding to the network's certainty on whther the agent should move left, right, up, or down given its current positioning. The snake agent then moves in the highest-certainty direction, and the reinforcement algorithm either rewards or punishes the agent based on the action taken. The agent learns how to play the game as this process is repeated for a number of specified training steps.

Even though our project diverged to focus on RL methods rather than genetic algorithms, a comparison between agents trained with the genetic algorithm and RL methods can be found in the [Appendix section](#7-appendix).

#### 4c. Datasets
As a result of this project centering on RL techniques and neuroevolution, we do not utilize large datasets and supervised learning to train snake agents. Rather, we are rewarding the agent for "good" actions and punishing it for "bad" ones through a set of reward functions that we specify (see [section 5c.](#5c-reward-structures) for details). These reward functions inform RL algorithms of "good" and "bad" moves as the agent is trained, and allows the algorithms to automatically adjust the NN's initially randomized weights and fine-tune the snake agents to learn the desired behavior.

Thus, we will not require an initial dataset for this project. Rather, we will generate data through simulation and experiment with how different networks perform.

#### 4d. Analysis
In terms of analysis and assessment, we have altered the original AI for Snake Game repository to provide us with cleaned data describing the performance of the genetic algorthm’s training. The performance data included highscore, average score, and median score for each generation, as well as metrics for the fitness of the network. 

Our updated repository leveraging OpenAI Gym for RL includes a script which trains different snake agents and also provides performance metrics. The performance metrics given here are simply games completed while testing, high score, average score, and median score.

## 5. Discussion
Over the course of the project we ran 54 total experiments to improve the overall performance of the snake where each experiment consisted of 100,000,000 training steps and 100,000 testing steps. Each experiment changed one or more aspects of how the training worked. In the following sections we explain our learnings through these experiments. Then we will describe the final parameters that we used to achieve a high score of 23 and median score of 18.

###### *Note 1: The raw data for results discussed below can be found [here](https://docs.google.com/spreadsheets/d/1yNhVSQuhbs6fgsL-IMh18bhgLSnmTLftfySUerWKHOE/edit?usp=sharing).*

###### *Note 2: We were unable to complete PPO (with border) training by December 17th, 2021. When these results are available, this page will be updated with relevant data and analysis.*

#### 5a. Board Size

| 5x5 Board | 10x10 Board |
|:---------:|:-----------:|
| ![5x5 Board](https://github.com/davidd-55/sNNake/blob/main/media/5x5_board_representation_compressed.png?raw=true) | ![10x10 Board](https://github.com/davidd-55/sNNake/blob/main/media/10x10_board_representation_compressed.png?raw=true) |

We experimented with two different board sizes: a 10x10 board and a 5x5 board. As one might expect, the snake was able to learn signicantly more quickly on a smaller board. On a 5x5 board, there are 25 pixels that make up the board representation. On a 10x10 board, there are 100 pixels. Thus on a 10x10 board, the snake needs much more time to get a deeper handle on how the game works. It takes longer for the snake to randomly find a fruit and it takes longer for the snake to randomly run into a wall.

We achieved the aggregate results below by training snake agents using the DQN algorithm and 9 variations of reward strctures for both 5x5 and 10x10 game board sizes for a total of 18 experiments.

#### Average scores across different board sizes

|       | Mean Score | Median Score | High Score |
|-------|------------|--------------|------------|
| 5x5   | 3.6241     | 3.667        | 11.8889    |
| 10x10 | 0.9604     | 1.111        | 4.7778     |

#### 5b. Board Representation
We had two different ways of representing the board environment (a 2-dimensional array of integers) to the reinforcement algorithm: a board with a specified border and a board without such a border. 

Border Not Represented             |  Border Is Represented 
:-------------------------:|:-------------------------:
![Border Not Represented](https://user-images.githubusercontent.com/19896216/144362121-37ca6e39-2698-4f48-95cb-16f3c3a31541.png)  |  ![Border Represented](https://user-images.githubusercontent.com/19896216/144362184-c0061707-abb6-443e-83f6-a9c8ac35f123.png)

To test which representation worked better, we ran 18 experiments on a 5x5 game board (9 using DQN algorithm and 9 using the A2C algorithm) where the border was **not** represented followed by the same 18 experiments where the border **was** represented. We hypothesized that a representation of the border would allow the model to get a more accurate understanding of the state at which the snake fails. The average of the mean scores, the median scores, and the high scores for each experiment can be found in the following table: 

#### Average scores across different board representations

|                        | Mean Score | Median Score | High Score |
|------------------------|------------|--------------|------------|
| border represented     | 4.362      | 4.667        | 12.778     |
| border not represented | 4.527      | 4.778        | 12.889     |

To our surprise, the border representation resulted in a slight disadvantage across mean, median, and high score averages. More data would be required to determine if this is statistically significant. This finding directly contradicted our hypothesis. 

Interestingly, when viewing our results by algorithm used, we found that DQN training with border representation performed better than the same training without a border representation, yet A2C training yielded the opposite result. 

#### Average scores across different board representations and algorithms

|     | Border Represented | Mean Score | Median Score | High Score |
|-----|--------------------|------------|--------------|------------|
| DQN | FALSE              | 3.624      | 3.667        | 11.889     |
| DQN | TRUE               | 4.177      | 3.667        | 12.444     |
| A2C | FALSE              | 5.431      | 5.889        | 13.889     |
| A2C | TRUE               | 4.955      | 5.667        | 13.111     |

Even though A2C training without a border representation contradicted our expectations, in general it outperformed DQN which is reiterated in [section 5d](#5d-algorithm).

#### 5c. Reward Structures
Our experiments used 9 custom reward structures to define whether to "punish" or "reward" snake agents at each step in the training process. Details of each reward structure, denoted *RS-A* through *RS-I*, can be found below:

- *RS-A:* "Kill after 10 idle moves"
    - We reward the snake +1 for consuming fruit and punish it -1 for hitting a wall. We kill the snake after 10 moves of failing to find and consume a fruit.
- *RS-B:* "Kill after 30 idle moves"
    - Same as *RS-A*, but kill the agent after 30 moves rather than 10.
- *RS-C:* "Kill and punish after 10 idle moves"
    - We reward the snake +1 for consuming fruit and punish it -1 for hitting a wall. We kill and punish the snake -1 after 10 moves of failing to find and consume a fruit.
- *RS-D:* "Kill and punish half as much after 10 idle moves"
    - Same as *RS-C*, but kill and punish the agent -.5 rather than -1 after 10 idle moves.
- *RS-E:* "Kill and punish one tenth as much after 10 idle moves"
    - Same as *RS-C*, but kill and punish the agent -.1 rather than -1 after 10 idle moves.
- *RS-F:* "Kill and punish after 30 idle moves"
    - We reward the snake +1 for consuming fruit and punish it -1 for hitting a wall. We kill and punish the snake -1 after 30 moves of failing to find and consume a fruit.
- *RS-G:* "Kill and punish half as much after 30 idle moves"
    - Same as *RS-F*, but kill and punish the agent -.5 rather than -1 after 30 idle moves.
- *RS-H:* "Kill and punish one tenth as much after 30 idle moves"
    - Same as *RS-F*, but kill and punish the agent -.1 rather than -1 after 30 idle moves.
- *RS-I:* "Punish one tenth for inactivity"
    - We reward the snake +1 for consuming fruit and punish it -1 for hitting a wall. We punish it -.1 for not doing anything, i.e. not consuming a fruit or colliding with a wall.

We conducted a total of 27 experiments in an effort to conclude which of the 9 reward structures worked the best. For each reward structure, we experimented its performance when paired with each of the DQN, A2C, and PPO algorithms (5x5 board, no border representation) and averaged the results in the table below:

#### Average scores across different reward structures

|        | Mean Score | Median Score | High Score |
|--------|------------|--------------|------------|
| *RS-A* | 4.335      | 4.667        | 13.000     |
| *RS-B* | 6.407      | 8.167        | 16.000     |
| *RS-C* | 4.817      | 5.667        | 12.667     |
| *RS-D* | 4.999      | 5.667        | 13.000     |
| *RS-E* | 4.956      | 5.333        | 13.333     |
| *RS-F* | 7.074      | 7.667        | 16.333     |
| *RS-G* | 8.843      | 11.333       | 17.667     |
| *RS-H* | 8.591      | 10.333       | 18.333     |
| *RS-I* | 3.096      | 3.000        | 7.000      |

The table demonstrates that *RS-G* and *RS-H*, i.e. killing and punishing the snake -.5/-.1 for 30 idle moves, are the best of the tested reward structures. Interestingly, *RS-G* tended to achieve slightly better mean/median scores, whereas *RS-H* achieved a slighty greater average high score.

#### 5d. Algorithms
Using [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/#), we were quite easily able to see how different RL algorithms affected results. ] In the end, we found that the stable-baseline3 implementation of PPO performed the best.

##### Average scores across different RL algorithms

|     | Mean Score | Median Score | High Score |
|-----|------------|--------------|------------|
| DQN | 3.624      | 3.667        | 11.889     |
| A2C | 5.431      | 5.889        | 13.889     |
| PPO | 8.650      | 11.056       | 16.667     |

The above table shows the averaged scores for the different RL algorithms. We performed 27 total experiments: 9 identical ones each for the DQN, A2C, and PPO algorithms. Each set of 9 experiments differed in reward structure employed, and all experiments leveraged a non-border representation of the game board.

#### 5e. Best Snake
Using all of the above learnings, we found that the best snake that we could train in terms of high score had the following traits:
* Algorithm: PPO
* Reward Structure: *RS-H*
* Border Represented: FALSE (No border represented = TRUE equivalent available)

Similarly, the best snake with regards to mean and median scores of 12.858 and 18, respectively, only differed in that it leveraged *RS-G*. This observation reflects the findings from the table in [section 5c](#5c-reward-structures).

The first snake achieved a high score of 23 on a 5x5 board, which is only 2 points shy of the maximum score of 25. Moreover, the version which achieved a mean score of 12.858 and a median score of 18 quickly learned how to efficiently play the game. 

| Untrained 5x5 sNNake | Trained 5x5 sNNake |
|:--------------------:|:------------------:|
| ![Untrained 5x5 sNNake](https://github.com/davidd-55/sNNake/blob/main/media/5x5_dumb_snake_compressed.gif?raw=true) | ![Trained 5x5 sNNake](https://github.com/davidd-55/sNNake/blob/main/media/5x5_smart_snake_compressed.gif?raw=true) |

We hypothesize that these snakes performed the best due to being trained on the PPO algorithm which routinely outperformed DQN and A2C implementations (see [section 5d](#5d-algorithms)). Additionally, reward structures G and H tend to coorespond to higher mean/median and high scores, respectively (see [section 5c](#5c-reward-structures)). We should note that a version of these agents trained with the PPO algorithm and a board representation containing a border is not currently available. So, we cannot conclude whether our current best snakes are better than identical versions trained on a board containing a border representation.

## 6. Ethics

#### 6a. General Considerations

As discussed above, reinforcement algorithms, similar to the algorithms we use in training our agent to play SS, utilize rewards mechanisms to drive behavior. Humans define these rewards, and in general, this can lead to unexpected and detrimental behavior especially when the desired behavior is very high stakes. For example, if the military decided to train a robotic agent using a reinforcement algorithm to seek out adversaries in a warzone, it would be incredibly hard to accurately define that “reward” and of course, this could potentially lead to the agent accidentally flagging civilians as adversaries. 

Thankfully, our project is quite removed from the physical world and our agent is not being trained with any real-world data; it is simply learning how to play a video game and our exploration is in the pursuit of improving our own understanding of reinforcement algorithms. Nonetheless, it is almost always the case that innocuous technologies can be reintroduced into real-world contexts, potentially leading to harsh ramifications. It is with this in mind that we would like to formally state that the research and experiments conducted in this project are in no way intended to be repurposed for harm of any kind.

We are only at the moment experimenting with different NNs for learning to play SS, and the ethical considerations are fairly generic and would apply to any project involving reinforcement algorithms. However, given that our project can be expanded in many directions - we or another developer could start to experiment with changes to the game itself and may come across more specific ethical considerations. For example, one idea that might be explored is training snake agents in different environments by changing the shape and size of the board and/or introducing obstacles. 

Returning to our war-machine example, we would not want our success in training an obstacle-weary snake agent to translate into the development of an obstacle-weary war-machine. Or for another example, we are interested in experimenting with training multiple snake agents to play in the same game with the hopes of observing learned collaboration. We would also not want success in this area to contribute to the development of a collaborating army of war-machines!

#### 6b. RL and Biology

Another completely different dimension of ethical concerns arise from considering the relevance of our project to biology. After all, our project centers around training agents to survive and adapt in an environment with particular constraints. We are interested in trying to breed successful snake agents and select behavior that leads to success. While this project borrows language from biology and loosely simulates a kind of evolution, we acknowledge that the development of snake behavior in this project does not necessarily apply to the development of human or other animal behaviors. Any non-trivial connection made between the two should excite major scrutiny.

#### 6c. Doing No Harm with RL Methods

When designing general RL approaches to gamified or other types of problems, we believe that it is important to consider a few important factors. We believe it is important to attempt to isolate biases in environmental factors. For instance, a robot trained to navigate a general environment with different biases corresponding to the color or size of obstacles could correspond to different behavior when the robot approaches humans with varying physical characteristics. Additionally, we have already seen RL algorithms negatively impact human psychological health with recommendation algorithms from companies such as Meta, YouTube, and Netflix. When designing learning algorithms to address a wide range of applications, it is crucial to consider the physical and non-physical impacts of those algorithms when they are deployed in the world.

## 7. Reflection

#### 7a. General Reflections

In terms of what we would have done differently, really our only regret is that we didn’t find OpenAI Gym earlier. The OpenAI Gym toolkit abstracted away most of the complexities of RL, and entirely made it possible for us to experiment with RL. Before we found OpenAI Gym, we spent too long trying to parse the intricacies of genetic algorithms to see if we could base our experiments solely on AI for Snake Game, once we concluded that this was impractical, we again spent precious time trying to get an intimate understanding of RL when it wasn’t needed. If we had started our project with OpenAI Gym or had looked for RL libraries earlier, we would have had much more time to explore extensions of our work.

Regarding the continuation of our work, we see four main topics we would have liked to explore in further depth: reward functions, game modifications, hyperparameter tuning, and community contributions.

#### 7b. Future Explorations in Reward Functions

While we created and tested many different reward functions, we anticipate there are a few ways we could improve the current reward functions. One isolated example is that we created a reward structure that rewarded the snake agent for reducing its Manhattan distance from the fruit and this reward structure performed very poorly, teaching the snake to only move back and forth. However, we realized recently that this reward structure could better teach the snake if there were a punishment for increasing the snake’s Manhattan distance from the fruit, this way the undesired fidgeting back and forth would not be rewarded but rather would be a net neutral state. However, while in that case we could have created a more intelligent reward structure, in general we also could have combined our existing functions to create new, more robust reward functions. We would have liked to explore both of these avenues for improvement if we had more time.

#### 7c. Future Explorations in Game Modifications

Regarding game modifications, we were hoping to have this be an area of exploration for our project with the primary modifications in mind being to add bombs and to change the shape of the board. However, there was so much coding to be done just to get our RL working and to experiment with reward functions that we could not spare the time to implement game modifications. Also, our RL controlled snake agents only recently began performing at a fairly high level, so it would not have made much sense to prematurely throw hurdles at our underperforming snake.

#### 7d. Future Explorations in Hyperparameter Tuning

When researching the different RL algorithms available with OpenAI Gym, we realized that very much like the NNs we have seen in class, RL algorithms also take hyperparameters. However, there were already so many variations of reward functions and game parameters to test in our experiments that we could not rationalize trying to tweak the different RL algorithm hyperparameters. We also have a very limited understanding of RL algorithms in general, so we would not have understood many of the hyperparameters available for us to tweak. Thus, we simply used OpenAI Gym’s default RL algorithms, however in the future this would be a great area to explore to hopefully increase training performance. 

#### 7e. Future Explorations in the AI Gaming Community

Finally, our work could provide numerous valuable contributions to the AI gaming community. By making our work highly searchable, we could aid other research groups hoping to study similar problems. It took us a while to find OpenAI Gym, however with more OpenAI Gym projects out there, maybe other groups would be able to find the toolkit faster than we did. Moreover we could try to connect our repo to the OpenAI Gym repo adding Snake as yet another example of a game able to be played with the toolkit. This would demonstrate OpenAI Gym’s vast versatility. Lastly, we could create a pull request to add our changes to the AI for Snake Game repo. We were very careful to preserve backwards compatibility with the original genetic algorithm learning paradigm, so we could give back to the repo which lended us its game infrastructure to stoke further discussion and inspire more work to be done in this area of inquiry.

## 8. Appendix

#### 8a. Genetic Algorithm Implementation

Craig Haber's AI for Snake Game leverages the use of a neuroevolution-based NN composed of a separate genetic algorithm and NN. This functions by training snake agents via a 4 layer (1 input, 2 hidden, 1 output) NN. 
    
At the end of a training epoch, the set of *n* agent results is fed to the genetic algorithm which improves agent performance through the following four methods (note that Fitness Calculations occur once per epoch, and Selection, Crossover, and Mutation are performed sequentially *n*/2 times):

- Fitness Calculations:  Calculations that assign scores to each agent based on its performance during the training epoch
- Selection: The algorithm selects 2 high-performing agents to be used for Crossover
- Crossover: The algorithm mixes characteristics of each relatively successful parent agent to create a new child agent
- Mutation: The algorithm randomly mutates aspects of the child agent's characteristics in order to prevent training stagnation

After the above steps are performed, the GA creates a set consisting of the top 50% of agents from the initial population and the newly generated child population and propogates them to the following training epoch. This process repeats for a specified number of epochs that each contain gameplay for a specified number of unique agents.

A basic overview of network inputs and outputs is described below, and further implementation details can be found on the original *AI for Snake Game* [project page](https://craighaber.github.io/AI-for-Snake-Game/website_files/index.html).

- Genetic Algorithm NN Input Details:
    - The [Manhattan distance](https://xlinux.nist.gov/dads/HTML/manhattanDistance.html) between the snake agent's head and the game board fruit
    - Available space on the grid in each of the four available directions (left, right, up, down) from the snake agent's head
    - The current length of the snake agent

- Genetic Algorithm NN Ouput Details: 
    - This network has four outputs (representing the four available movement directions) ranging in value from 0 to 1
    - The snake agent will move in the direction of the output with the highest associated value

#### 8b. Genetic Algorithm Discussion
As discussed above, we briefly explored the initially implemented genetic algorithm to train an agent but decided to focus on RL algorithms instead. That said, before completely abandoning the effort, we trained and tested [Craig Haber's genetic algorithm snake agent](https://craighaber.github.io/AI-for-Snake-Game/website_files/index.html) for 1,000 generations with the following results:

![Average Score vs  Generation](https://user-images.githubusercontent.com/19896216/137426767-8fcf979b-9b71-4596-8260-bee82b7c06da.png)

The algorithm was run for 3 days 4 hours and 44 minutes on a 10x10 board. By the end of the training, the agent's average score was 4.95. 

In contrast, we trained one of our best performing RL algorithms (PPO, reward structure G) on a 10x10 board for 6 days 4 hours and 16 minutes to achieve an average score of 6.60 after 100,000,000 training steps. 

Our RL-trained snake implementation achieved a 33% higher score at the cost of nearly twice as much training time when compared to the genetic algorithm-trained agent. Given the low amount of data available as a result of long training times and our project's focus shifting away from genetic algorithms, we cannot make a definite claim regarding which training method is better suited to the SS game. However, it is worth noting that both approaches were unable to perform well, even after days of training, on a 10x10 board version of the game.
