#**************************************************************************************
#snakeGameGym.py
#Module with the SnakeGameGym class that is instantiated in testTrainedAgents.py
#to observe the best agents that were trained with the reinforcement learning algorithm.
#*************************************************************************************

import numpy as np
from gym import spaces
import collections
from gym_snake.envs.snakeGame import SnakeGame
from gym_snake.envs.snake import Snake
import pygame


class SnakeGameGym(SnakeGame):
	"""
	Class framework to observe agents who were trained with the reinforcement algortihm to play the Snake Game.

	Inherits the SankeGame class that runs the Snake Game.
	"""

	def __init__(self, fps: int, use_pygame: bool = True):
		"""
		Initializes the SnakeGameGATest class.
		"""
		self.use_pygame = use_pygame
		self.move_map = {
			0: "left",
			1: "up",
			2: "right",
			3: "down",
		}
		
		self.width = 500
		self.height = 600
		self.grid_start_y = 100
		self.play = True
		self.restart = False
		self.fps = fps  # FIXME: remove fps since it doesn't seem to be doing anything
		self.rows = 10
		self.cols = self.rows
		self.snake = Snake(self.rows,self.cols)
		self.fruit_pos = (0,0)
		self.generate_fruit()
		self.score = 0
		self.high_score = 0	

		if self.use_pygame:
			self.win = pygame.display.set_mode((self.width, self.height))
			self.clock = pygame.time.Clock()


	def pos_on_board(self, pos):
		# If row index is less than 0 or greater than number of rows, pos is not on board
		if pos[0] < 0 or pos[0] >= self.rows:
			return False

		# If col index is less than 0 or greater than number of cols, pos is not on board
		if pos[1] < 0 or pos[1] >= self.cols:
			return False		

		# Otherwise, pos is on board
		return True

	def get_ga_network_inputs(self):
		head = self.snake.body[0]

		#Get the manhattan ditance of the fruit from the head if it moves in each direction
		dist_left_fruit = self.manhattan_distance(head[0],head[1]-1)
		dist_up_fruit = self.manhattan_distance(head[0]-1,head[1])
		dist_right_fruit = self.manhattan_distance(head[0],head[1]+1)
		dist_down_fruit = self.manhattan_distance(head[0]+1, head[1])

		#Calculate the space available for turning in each of the four directions, reduced by a constant factor
		constant = 20
		open_spaces_left = self.calc_open_spaces((head[0], head[1]-1))/constant
		open_spaces_up = self.calc_open_spaces((head[0]-1, head[1]))/constant
		open_spaces_right = self.calc_open_spaces((head[0], head[1]+1))/constant
		open_spaces_down = self.calc_open_spaces((head[0]+1, head[1]))/constant

		#Get the length of the snake
		length = self.score + 1

		return np.array([dist_left_fruit, dist_up_fruit, dist_right_fruit, dist_down_fruit,  open_spaces_left, open_spaces_up, open_spaces_down, open_spaces_right, length], dtype=np.float32)

	def get_board(self) -> np.ndarray:
		"""
		Uses sefl.rows, self.cols, self.snake, and self.fruit_pos in order
		to create a list representation of the board
		0 is empty space
		1 is space with fruit in it
		2 is space with snake body in it
		3 is space with snake head in it
		"""
		# Initializes empty board
		board = np.zeros([self.rows, self.cols], dtype=int)

		# Add Fruit
		fruit_row = self.fruit_pos[0]
		fruit_col = self.fruit_pos[1]
		board[fruit_row][fruit_col] = 1

		# Add Snake to Board
		for i in range(len(self.snake.body)):
			pos = self.snake.body[i]
			
			# If body position is outside of the board, do not add this to board representation
			if not self.pos_on_board(pos):
				# FIXME: figure out a better way to represent this condition
				pass

			# Add Snake Head
			elif i == 0:
				board[pos[0]][pos[1]] = 3
			
			# Add rest of Snake Body
			else:
				board[pos[0]][pos[1]] = 2

		return board

	def move_snake(self, action: spaces.Discrete(4)) -> None:
		"""
		Function that moves the snake on the board in one of four possible directions
		using a discrete 4-item action space as input.
		"""
		direct = self.move_map[action]

		self.snake.directions.appendleft(direct)
		if len(self.snake.directions) > len(self.snake.body):
			self.snake.directions.pop()

		self.snake.update_body_positions()

	def respond_to_fruit_consumption(self) -> int:
		"""
		Function that extends a snake, generates new snake tail block and fruit,
		and updates/returns the new score
		"""
		#Add the new body square to the tail of the snake
		self.snake.extend_snake()
		#Generate a new fruit in a random position
		self.generate_fruit()
		#Update score
		self.score += 1

		return self.score

	def manhattan_distance(self, y_head, x_head):
		"""Function to calculate the manhattan distance between the fruit and the snake's head

		Arguments:
			y_head: The row in the grid of the snake's head.
			x_head: The column in the grid of the snake's head.

		Returns:
			The manhattan distance between the fruit and the snake's head.
		"""
		return abs(self.fruit_pos[0] - y_head) + abs(self.fruit_pos[1] - x_head)
	
	def calc_open_spaces(self,start_pos):
		"""Function to calculate the number of open spaces around the snake 

		An open space is a space that the snake can reach without being blocked off by
		the wall or its own body.

		Arguments:
			start_poistion: A tuple in (row,column) format representing a position of the snake's head

		Returns:
			An integer of how many open spaces are available.
		"""
		open_spaces = 0

		start_y = start_pos[1]
		start_x = start_pos[0]

		#If the start position is in the snake's body or out of bounds
		if start_pos in self.snake.body or (start_x < 0 or start_x >= self.cols or start_y < 0 or start_y >= self.rows):
				#no open spaces
				return 0

		#Breadth first search is used

		#Create a set to represent th visited spaces
		visited = set([start_pos])
		#Create a queue to keep track of which spaces need to be expanded
		queue = collections.deque([start_pos])

		#While there are still unvisited open spaces to search from
		while len(queue) > 0:

			cur = queue.popleft()

			possible_moves = self.get_possible_moves(cur)

			for move in possible_moves:
				if move not in visited:

					visited.add(move)

					#if the move is an open space
					if move not in self.snake.body:
						open_spaces +=1
						#add the open space to the queue for further searching
						queue.append(move)

		return open_spaces

	def get_possible_moves(self,cur):
		"""Function to get all the possible adjacent moves from a position.

		The function is called from calc_open_spaces() during the breadth first search.

		Arguments:
			cur: A tuple in (row,column) format representing the position
			to get the next possible moves from.

		Returns:
			A list containing (row,column) tuples of all the possible adjacent moves.
		"""

		adjacent_spaces = [(cur[0], cur[1]-1), (cur[0]-1,cur[1]), (cur[0], cur[1]+1), (cur[0]+1, cur[1])]
		possible_moves = []
		for move in adjacent_spaces:
			move_y = move[1]
			move_x = move[0]
			#If the move is not out of bounds
			if move_x >= 0 and move_x < self.cols and move_y >= 0 and move_y < self.rows:
					possible_moves.append(move)
		return possible_moves

	def check_collisions(self) -> int:
		"""
		Function that consecutively calls all the functions that detect collisions
		Returns a reward based on these collisions
		"""
		fruit_collision = self.check_fruit_collision()
		wall_collision = self.check_wall_collision()
		body_collision = self.check_body_collision()		
		
		if fruit_collision:
			return 1
		elif wall_collision or body_collision:
			return -1
		else:
			return 0

	def check_fruit_collision(self) -> bool:
		"""
		Function that detects and handles if the snake has collided with a fruit.
		"""
		#If we found a fruit
		if self.snake.body[0] == self.fruit_pos:
			return True
		
		return False

	def check_wall_collision(self) -> bool:
		"""
		Function that checks and handles if the snake has collided with a wall.
		"""
		#Only need to check the colisions of the head of the snake
		head = self.snake.body[0]  # TODO: create head state variable
		head_y = head[0]
		head_x = head[1]

		#If there is a wall collision, game over
		if head_x == self.cols or head_y == self.rows or head_x < 0 or head_y < 0:
			return True
		
		return False

	def check_body_collision(self) -> bool:
		"""
		Function that checks and handles if the snake has collided with its own body.
		"""
		if len(self.snake.body) > 1:
			#Only need to check the colisions of the head of the snake
			head = self.snake.body[0]
			body_without_head = self.snake.body[1:]

			#Check for head collision with rest of snake body
			if head in body_without_head:
				return True

		return False
