"""
BASIC SNAKE GAME USING PYGAME

Some additional features I added :
1. A background checkered board 
2. A score counter in the bottom right to keep track during the game too
3. A check to make sure fruit isnt spawned at any current block occupied by snake
4. Best score Counter

"""
import pygame
import ctypes
import sys, random
from pygame.math import Vector2
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Direction():
    RIGHT = Vector2(1, 0)
    LEFT = Vector2(-1, 0)
    UP = Vector2(0, -1)
    DOWN = Vector2(0, 1)

class FRUIT:
    def __init__(self,snake_body,screen):                  #using snake_body as an argument to use the self.body from snake class in fruit class
        self.randomize(snake_body)
        self.screen = screen

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x*cell_size),int(self.pos.y*cell_size),cell_size,cell_size)
        pygame.draw.rect(self.screen,(126,166,114),fruit_rect)
    def randomize(self,snake_body):
        while True :
            self.x = random.randint(0,cell_number-1)
            self.y = random.randint(0,cell_number-1)
            new_pos = Vector2(self.x,self.y)
            if new_pos not in snake_body :          #imp feature imo that both resources missed, where fruit shouldnt spawn where snake's body is present
                self.pos = new_pos                  #this makes game less confusing when snake gets long enough
                break
class SNAKE:
    def __init__(self,screen):
        self.direction=Direction.RIGHT
        self.body = [Vector2(cell_number//4,cell_number//2),Vector2(cell_number//4-1,cell_number//2),Vector2(cell_number//4-2,cell_number//2)]
        self.new_block = False
        self.screen=screen

    def draw_snake(self):
        for block in self.body :
            snake_rect = pygame.Rect(int(block.x*cell_size),int(block.y*cell_size),cell_size,cell_size)
            pygame.draw.rect(self.screen,(pygame.Color("red")),snake_rect)
    def move_snake(self):
        if self.new_block==True: 
            body_copy = self.body[:]
            body_copy.insert(0,body_copy[0]+self.direction)
            self.body = body_copy
            self.new_block = False
        else : 
            body_copy = self.body[:-1]
            body_copy.insert(0,body_copy[0]+self.direction)
            self.body = body_copy

    def add_block(self):
        self.new_block = True

class MAIN:
    def __init__(self):
        self.screen = pygame.display.set_mode((cell_size*cell_number,cell_size*cell_number))      
        ctypes.windll.user32.SetForegroundWindow(pygame.display.get_wm_info()['window'])
        self.clock = pygame.time.Clock()
        self.reset_game()      

    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()
    
    def draw_elements(self):
        self.draw_grass()
        self.fruit.draw_fruit()
        self.snake.draw_snake() 
              
    def draw_grass(self):
        grass_color = (167, 209, 60)
        for row in range(cell_number):
            if row%2 == 0 :
                for col in range(cell_number):
                    if col%2 == 0:
                        grass_rect = pygame.Rect(col*cell_size,row*cell_size,cell_size,cell_size)
                        pygame.draw.rect(self.screen,grass_color,grass_rect)
            else :
                for col in range(cell_number):
                    if col%2 != 0:
                        grass_rect = pygame.Rect(col*cell_size,row*cell_size,cell_size,cell_size)
                        pygame.draw.rect(self.screen,grass_color,grass_rect)    

    def check_collision(self):
        if self.fruit.pos==self.snake.body[0]:
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
            self.reward = 10
            self.score += 1
    def check_fail(self):
        if self.snake.body[0].x not in range(0,cell_number) or self.snake.body[0].y not in range(0,cell_number):
            self.game_over()
        for block in self.snake.body[1:]:
            if self.snake.body[0] == block :
                self.game_over()
        if self.frame_iteration > 100 * len(self.snake) :
            self.game_over()
    def game_over(self):
        # my_font = pygame.font.SysFont('times new roman', 50, bold=True)
        # small_font = pygame.font.SysFont('times new roman', 30)
        self.over = True
        self.reward = -10
        self.score = len(self.snake.body) - 3
          # still used in rendering
        return self.reward,self.over,self.score
        self.reset_game()
       
    def reset_game(self):
        self.snake = SNAKE(self.screen)       # reset snake position, direction, and body
        self.fruit = FRUIT(self.snake.body,self.screen) 
        self.frame_iteration = 0
        self.score = 0
    
    def move(self,action):
        SCREEN_UPDATE = pygame.USEREVENT                
        pygame.time.set_timer(SCREEN_UPDATE,150)
        running = True
        
        self.frame_iteration += 1
        self.reward = 0
        self.over = False   #game_over bool in code
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir
        self.update()
        self.screen.fill((175, 215, 69))
        self.draw_elements()
        pygame.display.update()
        self.clock.tick(69)

        return self.reward,self.over,self.score
    
class Linear_QNet(nn.Module):
    """
    For our case, input size would mean number of elements in state (11), and output size is no. of possible 
    actions (3).
    No activation function after output layer since we're computing raw q values and dont need probabilities
    """
    def __init__(self, input_size, hidden_size, output_size): #building the input, hidden and output layer
        super().__init__() #allows class to access all functions from nn.Module (the parent class)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): #this is a feed-forward neural net where x is current game state
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x        

    def save(self, file_name='model.pth'): #saving the model, later we can reuse the weights and biases if we want
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)          
class QTrainer:
    def __init__(self, model, lr, gamma): #initializing 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer
        self.criterion = nn.MSELoss() #loss function, use Huber (SmoothL1) if training is unstable

    def train_step(self, state, action, reward, next_state, done): #trainer
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: #if there 1 dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state) #using the Q=model predict equation above
        """
        rn pred has q values of each action for that given state. in the next line we create a  copy of pred
        and only update the q value of the action we plan on taking. we do so by applying the bellman equation
        to that specific q value.
        """
        target = pred.clone() #using Qnew = r+y(next predicted Q) as mentionned above
        for idx in range(len(done)): #if game over or episode ends
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad() #calculating loss function
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001





pygame.init()
cell_size = 30
cell_number = 15

score = 0


main_game = MAIN()
main_game.move()



#pygame.display.set_caption('Cool Snake Game')
#screen = pygame.display.set_mode((cell_size*cell_number,cell_size*cell_number))
#ctypes.windll.user32.SetForegroundWindow(pygame.display.get_wm_info()['window'])
#clock = pygame.time.Clock()
# while True : 
#     # draw all our elements
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             main_game.game_over()
#         if event.type == SCREEN_UPDATE:
#             main_game.update()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 if main_game.snake.direction.y != 1 :
#                     main_game.snake.direction = Vector2(0,-1)
#             if event.key == pygame.K_RIGHT:
#                 if main_game.snake.direction.x != -1 :
#                     main_game.snake.direction = Vector2(1,0)
#             if event.key == pygame.K_DOWN:
#                 if main_game.snake.direction.y != -1 :
#                     main_game.snake.direction = Vector2(0,1)
#             if event.key == pygame.K_LEFT:
#                 if main_game.snake.direction.x != 1 :
#                     main_game.snake.direction = Vector2(-1,0)                
        
#     main_game.screen.fill((175,215,69))
#     main_game.draw_elements()
    
    
#     pygame.display.update()
#     main_game.clock.tick(69)
