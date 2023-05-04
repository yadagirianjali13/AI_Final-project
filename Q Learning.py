

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

