"""
SNAKE GAME
"""

#Implementation of Reinforcement learning on a Snake Game 

"""
INSTALLATION PROCESS 

"""

![pytorch](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)
install pytorch from here: https://pytorch.org/
"""
Requirements mentioned in the text file are needed to be installed for the project 
"""
#command for installing the libraries

pip install -r requirements.txt


""" 
Code for to run the Game
"""

python main.py


## Configurations
All static settings are in settings.py
```python
from pygame import display, time, draw, QUIT, init, KEYDOWN, K_a, K_s, K_d, K_w
from random import randint
import pygame
from numpy import sqrt
init()

done = False
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

cols = 25
rows = 25

width = 600
height = 600
wr = width/cols
hr = height/rows
direction = 1

screen = display.set_mode([width, height])
display.set_caption("snake_self")
clock = time.Clock()


def getpath(food1, snake1):
    food1.camefrom = []
    for s in snake1:
        s.camefrom = []
    openset = [snake1[-1]]
    closedset = []
    dir_array1 = []
    while 1:
        current1 = min(openset, key=lambda x: x.f)
        openset = [openset[i] for i in range(len(openset)) if not openset[i] == current1]
        closedset.append(current1)
        for neighbor in current1.neighbors:
            if neighbor not in closedset and not neighbor.obstrucle and neighbor not in snake1:
                tempg = neighbor.g + 1
                if neighbor in openset:
                    if tempg < neighbor.g:
                        neighbor.g = tempg
                else:
                    neighbor.g = tempg
                    openset.append(neighbor)
                neighbor.h = sqrt((neighbor.x - food1.x) ** 2 + (neighbor.y - food1.y) ** 2)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.camefrom = current1
        if current1 == food1:
            break
    while current1.camefrom:
        if current1.x == current1.camefrom.x and current1.y < current1.camefrom.y:
            dir_array1.append(2)
        elif current1.x == current1.camefrom.x and current1.y > current1.camefrom.y:
            dir_array1.append(0)
        elif current1.x < current1.camefrom.x and current1.y == current1.camefrom.y:
            dir_array1.append(3)
        elif current1.x > current1.camefrom.x and current1.y == current1.camefrom.y:
            dir_array1.append(1)
        current1 = current1.camefrom
    #print(dir_array1)
    for i in range(rows):
        for j in range(cols):
            grid[i][j].camefrom = []
            grid[i][j].f = 0
            grid[i][j].h = 0
            grid[i][j].g = 0
    return dir_array1


class Spot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.camefrom = []
        self.obstrucle = False
        if randint(1, 101) < 3:
            self.obstrucle = True

    def show(self, color):
        draw.rect(screen, color, [self.x*hr+2, self.y*wr+2, hr-4, wr-4])

    def add_neighbors(self):
        if self.x > 0:
            self.neighbors.append(grid[self.x - 1][self.y])
        if self.y > 0:
            self.neighbors.append(grid[self.x][self.y - 1])
        if self.x < rows - 1:
            self.neighbors.append(grid[self.x + 1][self.y])
        if self.y < cols - 1:
            self.neighbors.append(grid[self.x][self.y + 1])


grid = [[Spot(i, j) for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        grid[i][j].add_neighbors()

snake = [grid[round(rows/2)][round(cols/2)]]
food = grid[randint(0, rows-1)][randint(0, cols-1)]
current = snake[-1]
dir_array = getpath(food, snake)
food_array = [food]

while not done:
    clock.tick(12)
    screen.fill(BLACK)
    direction = dir_array.pop(-1)
    if direction == 0:    # down
        snake.append(grid[current.x][current.y + 1])
    elif direction == 1:  # right
        snake.append(grid[current.x + 1][current.y])
    elif direction == 2:  # up
        snake.append(grid[current.x][current.y - 1])
    elif direction == 3:  # left
        snake.append(grid[current.x - 1][current.y])
    current = snake[-1]

    if current.x == food.x and current.y == food.y:
        while 1:
            food = grid[randint(0, rows - 1)][randint(0, cols - 1)]
            if not (food.obstrucle or food in snake):
                break
        food_array.append(food)
        dir_array = getpath(food, snake)
    else:
        snake.pop(0)

    for spot in snake:
        spot.show(WHITE)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j].obstrucle:
                grid[i][j].show(RED)

    food.show(GREEN)
    snake[-1].show(BLUE)
    display.flip()
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        elif event.type == KEYDOWN:
            if event.key == K_w and not direction == 0:
                direction = 2
            elif event.key == K_a and not direction == 1:
                direction = 3
            elif event.key == K_s and not direction == 2:
                direction = 0
            elif event.key == K_d and not direction == 3:
                direction = 1


#Add the windows 

A* Algorithm
```python

class Windows(Enum):

    
    W14 = (20, 20, 1, 1)
```

#TO RUN THE WINDOW

The following will run window 14
```python

if __name__ == "__main__":
    Game(14)
```

#ADDING PARAMETERS TO THE PROCESSESS
"""
In par_lev.json:
Add parameters for instance "world 1" where world is (20, 20, 1, 3) the following: 
first processor will have all these parameters, second the epsilon changes to 80 and graph name different and third parameters will use the 
default, look at the report for more info on default values and settings.py.
json
"""


import threading
import turtle
from turtle import *
from random import randrange
import random
from freegames import square, vector
import numpy as np
from pynput.keyboard import Key, Controller
import time


class SnakeGameEnvironment:


    MATRIX_SIZE = 620
    SCORE = 0

    def __init__(self, agent, numberOfObstacles= 25):
        self.keyboard = Controller()
        self.food = vector(0, -100)
        self.numberOfObstacles = numberOfObstacles
        self.obstacles = []
        self.snake = [vector(10, 20), vector(10, 10), vector(10, 0)]
        self.aim = vector(0, -10)
        self.direction = 'Down'
        self.agent = agent
        self.reward = 0
        self.spanObstacles()
        self.countscore = 0


    def getreward(self):
        return SnakeGameEnvironment.SCORE

    def spanObstacles(self):
        x, y = 0, 0
        for i in range(self.numberOfObstacles):
            while (True):
                x = randrange(-19, 19) * 10
                y = randrange(-19, 19) * 10
                if vector(x, y) not in self.snake:
                    break
            self.obstacles.append(vector(x, y))

    def change(self, x, y, direction):
        "To Change snake direction."
        if direction in self.getAvailableDirections():
            self.aim.x = x
            self.aim.y = y
            self.direction = direction

    def inside(self, head):
        "Return True if head inside boundaries."
        return -200 < head.x < 200 and -200 < head.y < 200

    def getCurrentState(self):
        head = self.snake[-1].copy()
        return head.x, head.y

    i = 0

    def coordinateSystemConverter(self, coord, to="normal"):
        x, y = coord
        mid = int(SnakeGameEnvironment.MATRIX_SIZE / 2)
        if to == "normal":
            x = mid + x
            y = mid - y
        else:
            x = mid - x
            y = mid + y
        return (x, y)

    def getReward(self, tempHead):
        if not self.inside(tempHead) or tempHead in self.snake:
            return -10
        if tempHead == self.food:

            return 50
        return -5

    def getNextRewardState(self):
        head = self.snake[-1].copy()

        head.move(self.aim)
        reward = []
        for direction in SnakeGameEnvironment.MOVABLE_DIRECTION:
            tempHead = self.transformMove(head, direction)
            reward.append(self.getReward(tempHead))
        return reward



    def move(self):
        "Move snake forward by one block."
        head = self.snake[-1].copy()
        head.move(self.aim)


        if not self.inside(head) or head in self.snake or head in self.obstacles:
            square(head.x, head.y, 9, 'blue')
            update()
            bye()
            SnakeGameEnvironment.i = 0
            self.reward = -100
            SnakeGameEnvironment.SCORE = SnakeGameEnvironment.SCORE + 100
            action = self.agent.Act(self.getState(head, self.food), self.MOVABLE_DIRECTION,
                                    self.reward, True)
            return
        else:
            self.snake.append(head)

            if head == self.food:

                i = 1
                while (True):
                    self.food.x = randrange(-20, 20) * 50
                    self.food.y = randrange(-20, 20) * 50
                    if self.food not in self.snake or self.food not in self.obstacles:
                        break
                self.reward = 500
                SnakeGameEnvironment.i = 0
            else:
                self.snake.pop(0)
                self.reward = -10
            action = self.agent.Act(self.getState(head, self.food), self.MOVABLE_DIRECTION,
                                    self.reward, False)
            self.direction = self.movableDirections(action, self.direction)
            self.aim = SnakeGameEnvironment.DIRECTIONS[self.direction]

        clear()

        for body in self.snake:
            square(body.x, body.y, 9, 'black')

        for obstacle in self.obstacles:
            square(obstacle.x, obstacle.y, 9, 'red')

        square(self.food.x, self.food.y, 9, 'green')
        # print(np.nonzero(self.rewardMatrix))
        # print(np.unique(self.rewardMatrix))
        # print(np.argwhere(self.rewardMatrix == SnakeGameEnvironment.BLACK).flatten())
        update()
        ontimer(self.move, 1)
        SnakeGameEnvironment.i = SnakeGameEnvironment.i + 1

    def changeDirection(self, direction):
        directions = [Key.up, Key.right, Key.down, Key.left]
        self.keyboard.press(directions[direction])
        self.keyboard.release(directions[direction])

    DIRECTIONS = {'Right': vector(10, 0), 'Left': vector(-10, 0), 'Up': vector(0, 10), 'Down': vector(0, -10)}

    def setup(self):
        try:
            setup(SnakeGameEnvironment.MATRIX_SIZE, SnakeGameEnvironment.MATRIX_SIZE,
                  int(SnakeGameEnvironment.MATRIX_SIZE / 2) + 10, 0)
            hideturtle()
            tracer(False)
            listen()
            onkey(lambda: self.change(10, 0, 'Right'), 'Right')
            onkey(lambda: self.change(-10, 0, 'Left'), 'Left')
            onkey(lambda: self.change(0, 10, 'Up'), 'Up')
            onkey(lambda: self.change(0, -10, 'Down'), 'Down')
            self.move()
            done()

            return None
        except turtle.Terminator:
            pass

    def getAvailableDirections(self):
        if self.direction == 'Right':
            return ['Right', 'Up', 'Down']
        elif self.direction == 'Left':
            return ['Left', 'Up', 'Down']
        elif self.direction == 'Up':
            return ['Right', 'Up', 'Left']
        else:
            return ['Right', 'Left', 'Down']

    MOVABLE_DIRECTION = ['GO_LEFT', 'GO_FORWARD', 'GO_RIGHT']

    def movableDirections(self, movingDirection, currentDirection):
        # currentDirection = self.direction
        if currentDirection == 'Right':
            if movingDirection == 'GO_LEFT':
                return 'Up'
            elif movingDirection == 'GO_RIGHT':
                return 'Down'
            else:
                return currentDirection
        elif currentDirection == 'Left':
            if movingDirection == 'GO_LEFT':
                return 'Down'
            elif movingDirection == 'GO_RIGHT':
                return 'Up'
            else:
                return currentDirection
        elif currentDirection == 'Up':
            if movingDirection == 'GO_LEFT':
                return 'Left'
            elif movingDirection == 'GO_RIGHT':
                return 'Right'
            else:
                return currentDirection
        else:
            if movingDirection == 'GO_LEFT':
                return 'Right'
            elif movingDirection == 'GO_RIGHT':
                return 'Left'
            else:
                return currentDirection


    def getNextSquareState(self, squareSpace):

        if not self.inside(squareSpace) or squareSpace in self.snake or squareSpace in self.obstacles:
            return -1
        if squareSpace == self.food:
            return 1
        return 0

    def SigNum(self, x):
        if x < 0:
            return -1
        if x > 0:
            return 1
        else:
            return 0

    def GetQuadrant(self, coord):
        (sign_x, sign_y) = (self.SigNum(coord[0]), self.SigNum(coord[1]))

        if sign_x == 0:
            qx = 0
        elif sign_x == 1:
            qx = 1
        else:
            qx = -1

        if sign_y == 0:
            qy = 0
        elif sign_y == 1:
            qy = 1
        else:
            qy = -1

        return (qx, qy)

    def TransformQuadrantBasedOnDirection(self, coord, d, directions):

        (x, y) = coord

        for direction in directions:
            if d == direction:
                if d == 'Left':  (x, y) = (y, -x)
                if d == 'Right': (x, y) = (-y, x)
                if d == 'Down':  (x, y) = (-x, -y)

        return self.GetQuadrant((x, y))

    def transformMove(self, head, movableDirection):
        tempHead = head.copy()
        direction = self.movableDirections(movableDirection, self.direction)
        tempHead.move(SnakeGameEnvironment.DIRECTIONS[direction])
        return tempHead

    def getState(self, head, food):
        square_description = []
        fruit = food.copy()
        head = head.copy()
        for direction in SnakeGameEnvironment.MOVABLE_DIRECTION:
            tempHead = self.transformMove(head, direction)
            square_description.append(self.getNextSquareState(tempHead))
        # print(head,fruit)
        head = self.coordinateSystemConverter(head)
        fruit = self.coordinateSystemConverter(fruit)
        # print(head,fruit)
        head = (head[0], -head[1])
        fruit = (fruit[0], -fruit[1])

        (x, y) = (fruit[0] - head[0], fruit[1] - head[1])
        (qx, qy) = self.TransformQuadrantBasedOnDirection((x, y),
                                                          self.direction, self.getAvailableDirections())
        mapped_state = (square_description[0], square_description[1],
                        square_description[2], qx, qy)
        return mapped_state
import _pickle as cPickle
import math
import random
import sys




class Agent():

    Q = {}

    def __init__(self, epsilon, trained_file="", gamma=0.9, alpha=0.8):
        self.gamma = gamma
        self.alpha = alpha
        if epsilon == -1.0:
            self.e = 0.5
        else:
            self.e = epsilon
        self.old_state = None
        self.old_action = None
        self.Q = {}
        self.N = {}
        self.count = 0

        if trained_file is not "":
            try:
                (self.e, self.count, self.Q) = cPickle.load(open(trained_file))
            except IOError as e:
                sys.stderr.write(("File " + trained_file + " not found. \n"))
                sys.exit(1)

        return

    def UpdateQ(self, state, action, state_, action_, reward, explore):
        # raise NotImplemented()
        if not state:
            return

        q = self.Q[state][action]
        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max(self.Q[state_].values())
            q += self.alpha * (reward + self.gamma * q_ - q)

        self.Q[state][action] = q

    def Act(self, state, actions, reward, episode_ended):
        self.count += 1

        # print(actions,rewards)
        if self.count == 10000:
            self.e -= self.e / 20
            self.count = 1000

        # epsilon-greedy
        if state not in self.Q:
            self.Q[state] = {}
            print("New snake arrived")
            for action in actions:
                self.Q[state][action] = 10

        # Explore
        fg = random.random()
        # print(fg , self.e)
        if fg < self.e:
            action = actions[random.randint(0, len(actions) - 1)]
            explore = True

        else:
            action = max(actions, key=lambda x: self.Q[state][x])
            explore = False

        if episode_ended:
            self.UpdateQ(self.old_state, self.old_action, None, None, reward,
                         explore)
        else:
            self.UpdateQ(self.old_state, self.old_action, state, action, reward,
                         explore)

        self.old_state = state
        self.old_action = action
        return action

    def WriteKnowledge(self, filename):
        fp = open(filename, "w")
        cPickle.dump((self.e, self.count, self.Q), fp)
        fp.close()
        return


if __name__ == '__main__':
    agent = Agent(0.5)


    for i in range(2000):
        game = SnakeGameEnvironment(agent)
        game.setup()
       # print(agent.Q)
        # input("hi")
        print("My score is: ", game.getreward())
    f = open("temp.pkl", 'wb')
    cPickle.dump(agent.Q, f)

Q learning algorithm

#Reinforcement Learning implementation 

Reinforcement learning (RL) is an area of machine learning where an agent aims to make the optimal decision in an uncertain environment in order to get the maximum cumulative reward. Since RL requires an agent to operate and learn in its environment, it's often easier to implement agents inside  simulations on computers than in real world situations. 

The main prupose of our project is to create We use the reinforcement learning mechanism on an existing video game (snake game). 
Here, The goal is to create a Reinforcement Learning agent to make an optimal decision in the closed environment to get the highest reward.
In this environment, it has incomplete information, and the state space of agents are quite large, We need to deal with these hurdles.



#HURDLES 
In this environment, it has incomplete information, and the state space of agents are quite large, We need to deal with these hurdles



## Environment and State Space

Representation for the snake game is in the form of n*n matrix. Each cell has a width and height of l pixels. A simplest approach is to use the pixel image to feel the agent. From the dimensions mentioned above, the state space would be of size |S| ∈ (n x n x l x l). While this method can work only for the smaller images but not for the larger images.As the size keep on increasing, we couldn't able to feed our agent properly. The size of the state space would reduce to |S| ∈ (n x n). This way of representing state is still not ideal as the state size increases exponentially as n grows. We will explore state reduction techniques in the next section.

## Action Space


Due to the simplistic nature of Snake there are only four possible actions that can be taken: up, down, left, and right. To speed up training and reduce backwards collisions, we simplified the actions down to three: straight, clockwise turn, and counter-clockwise turn. Representing the actions in this way is beneficial because when the agent 'explores' and randomly picks an action, it will not turn 180 degrees into itself. 

## Positive/Negative Rewards


The main reward of the game is when the snake eats food and increases its score. Therefore the reward is directly linked to the final score of the game, similar to how a human would judge its reward. As we will discuss later, we experimented with other positive rewards but ran into a lot of unexpected behaviour. With other positive rewards, the agent may loop infinitely or learn to avoid food altogether to minimize its length. We included additional negative rewards to give the snake more information about its state: collision detection , loop , empty cells, and close/mid/far/very_far from food.


# Methods and Models
---

A common RL algorithm that is used is Q-Learning which has been expanded to include neural networks with Deep Q-Learning methods. We decided that we could experiment with this new method that is gaining popularity and is used in previous research done with Atari games .

To begin our tests we first used PyGame to create our own Snake game with the basic rules of movement. The snake's actions  are simply to move forward, left, or right based on the direction its facing. The game ends if the snake hits itself or the wall. As it consumes food it grows larger. The goal is to get the snake to eat as much food as possible without ending the game.

After the game was created we created a Deep Q-Learning network using PyTorch. We created a network with an input layer of size 11 which defines the current state of the snake, one hidden layer of 256 nodes, and an output layer of size 3 to determine which action to take. Pictured below is a visual representation of our network.



Due to the discrete time steps (frames) of the game, we are able to calculate a new state for every new frame of the game. We defined our state parameters to be 11 boolean values based on the direction the snakes moving, the location of danger which we define as a collision that would occur with an action in the next frame, and the location of food relative to the snake. Our 3 actions (the output layer) are the directions of movement for the snake to move relative to the direction it's facing: forward, left, or right. 

The state at every time step is passed to the Q-Learning network and the network makes a prediction on what it thinks is the best action. This information is then saved in both short term and long term memory. All the information learned from the previous states can be taken from memory and passed to the network to continue the training process. 


# Experiments

We will look into the experiments and state how well the deep-q learning parameters are performing well and also first our agent is procceding with the untrained policy and later on we will be implementing a policy and training our agent with the parameters we have taken.


To reduce the randomness in our experiments. At the beginning of the experiment we will be taking 3 agents and computed the average of the 3 agents. Due to the slow processing of the average computation. we will be taking the trained set of data over 300 games
We will be presenting our results with a double plot. The top plot will be the 5 game moving average of the score and the bottom plot will be the highest score the agent achieved.

## No Training (Base Case)

The untrained agent moved around sporadically and without purpose. The highest score it achieved was only one food. As expected, there was no improvement in performance.

## Default Parameters
---

![Default](graphs/default.png)

We decided to set our default parameters as the following, and made changes to individual parameters to see how they would change the performance: 

```python
  Gamma = 0.9
  Epsilon = 0.4
  Food_amount = 1
  Learning_Rate = 0.001
  Reward_for_food (score) = 10
  collision_with_Wall = -10
  collision_with_itself = -10
  Snake_going_in_a_loop = -10
```
 
#GAMMA_VALUE_EXPERIMENT 
---

![Gamma](graphs/gamma.png)

We decided to test gamma values of 0.00, 0.50, and 0.99. A gamma of 0.00 means that the agent is only focused on maximizing immediate rewards. We assumed a gamma of 0 would be ideal because the game of Snake is not a game where you have to think very far in the future. The results show that the best performance was with a gamma of 0.50 which showed much better performance than the other two. We are unsure why a gamma of 0.99 performed so badly. Our default value of gamma=0.90 performed the best. This demonstrates that it is necessary to fine tune gamma to balance the priority of short term reward vs long term.

#EPISLON_VALUE_EXPERIMENT 


Our epsilon decays by 0.005 every game until it reaches 0. This allows our agent to benefit from exploratory learning for a sufficient amount of time before switching to exploitation.
 
We wanted to test how changing the balance between exploration and exploitation impacts the performance. We decided to try no exploration (epsilon = 0), a balance of both (epsilon = 0.5), and the largest amount of exploration at the beginning (epsilon = 1).

As seen by the graph above, an epsilon of 0 performs poorly because without exploration the agent cannot learn the environment well enough to find the optimal policy. An epsilon of 0.5 provides an even balance of exploring and exploitation which greatly improves the learning of the agent. An epsilon of 1 maximizes the amount of time the agent explores in the beginning. This results in a slow rate of learning at the beginning but a large increase of score once the period of exploitation begins. This behaviour proves that high epsilon values are needed to get a higher score. To conclude, epsilon values of 0.5 and 1 both seem to be more performant than the default of 0.4.

#REWARDS


In this experiment we decided to change the immediate rewards. Our immediate rewards were (S) score for eating food, (C) collision with the wall or itself, and (L) moving in a loop. An interesting result we came across is that having a large difference between positive and negative rewards negatively affects the performance of the agent. This may be because the agent will learn to focus on the larger of the negative or positive rewards therefore making rewards of equal magnitude be more performant. We also found that having rewards that are small in scale do better than rewards that are large in scale. The best performance we found was with rewards of C=-5, L=-5, and S=5. Rewards of C=-1, L=-1, and S=1 performed very similarly to the previous agent. Larger rewards of 30 and -30 performed much worse. The performance of rewards C=-5, L=-5, and S=5 did slightly better than default.
 
#Learning Rate 


Changing the learning rate impacts the way our agent finds an optimal policy and how fast it learns. We found that a learning rate of 0.05 was too high since it performed similar to our untrained agent. This poor performance is likely because the agent was skipping over the optimal by taking too large a step size. This means that the agent moves too quickly from one suboptimal solution to another and fails to learn. We noticed a strong correlation between lower learning rates and higher scores. Our best performing agents had the lowest learning rates with lr = 0.0005 performing better than our default of lr = 0.001.

#OPTIMALAGENT

The performance of our optimal agent is slightly better than the default. This is because our default parameters were similar to the optimal parameters from our experiments. Further experimentation would allow for more finetuning of parameters to increase performance.
	From our experiments we found that the learning rate, epsilon, gamma, and immediate rewards were the parameters that had the biggest impact on performance. The experiments with direction, distance, and food generation were detrimental to performance and are not parameters that would help with the optimal performance.


Based on our experiments we decided to take the optimal values we found to see how the agent would perform over a 1000 games compared to the default. The optimal parameters we used were:
```python
Gamma = 0.9
Epsilon = 1
Food amount = 1
Learning Rate = 0.0005
Reward for food (score) = 5
Collision to wall = -5
Collision to self = -5
Snake going in a loop = -5
```

The performance of our optimal agent is slightly better than the default. This is because our default parameters were similar to the optimal parameters from our experiments. Further experimentation would allow for more finetuning of parameters to increase performance.

From our experiments we found that the learning rate, epsilon, gamma, and immediate rewards were the parameters that had the biggest impact on performance. The experiments with direction, distance, and food generation were detrimental to performance and are not parameters that would help with the optimal performance.


Above is a graph showing the high scores for each experiment for each parameter.
	
We combined the parameters that had the best impact on performance and used them as part of our optimal parameters. We found that small changes in the learning rate had the largest difference in performance. From the lowest to its highest result, the difference was a record of 79. The rewards had the second largest range, then epsilon, and then gamma. 

Our experiments were a starting point of looking at parameters that would impact Deep Q-Learning. Further research could be done to tune the main parameters to optimize the model further. 


#REFERENCE


https://www.researchgate.net/post/Reinforcement_learning_vs_heuristic_search
https://en.wikipedia.org/wiki/A*_search_algorithm
https://en.wikipedia.org/wiki/Q-learning
https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/
https://github.com/Karthikeyanc2/Autonomous-snake-game-with-A-star-algorithm-in-PYTHON
https://github.com/AndresRubianoM/Snake_DeepLearning
