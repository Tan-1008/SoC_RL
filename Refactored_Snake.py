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
import copy
import matplotlib.pyplot as plt
from IPython import display

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
            self.steps_since_last_fruit =0
    def check_fail(self):
        if self.snake.body[0].x not in range(0,cell_number) or self.snake.body[0].y not in range(0,cell_number):
            self.game_over()
        for block in self.snake.body[1:]:
            if self.snake.body[0] == block :
                self.game_over()
        if self.steps_since_last_fruit > MAX_STEPS_WITHOUT_FRUIT :
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
        self.steps_since_last_fruit =0
    
    def move(self,action):
        SCREEN_UPDATE = pygame.USEREVENT                
        pygame.time.set_timer(SCREEN_UPDATE,150)
        running = True
        
        self.frame_iteration += 1
        self.steps_since_last_fruit += 1
        self.reward = 0
        self.over = False   #game_over bool in code
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.snake.direction = new_dir
        
        self.update()
        self.screen.fill((175, 215, 69))
        self.draw_elements()
        pygame.display.update()
        self.clock.tick(69)
        if self.reward == 0:
            self.reward = -0.01

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
MAX_STEPS_WITHOUT_FRUIT = 500

class Agent:
    def __init__(self):
        self.main_game = MAIN()
        self.n_games = 0
        self.epsilon = 1  # randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995   
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  
        self.model = Linear_QNet(11, 256, 3) #input size, hidden size, output size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)     
    
    def is_collision(self,direction):
        
        head = self.main_game.snake.body[0]
        new_head = head + direction

        # Wall collision
        if new_head.x < 0 or new_head.x >= cell_number or new_head.y < 0 or new_head.y >= cell_number:
            return True

        # Self collision
        if new_head in self.main_game.snake.body[1:]:
            return True

        return False
    
    
    def get_state(self):        
        dir_l = self.main_game.snake.direction == Direction.LEFT
        dir_r = self.main_game.snake.direction == Direction.RIGHT
        dir_u = self.main_game.snake.direction == Direction.UP
        dir_d = self.main_game.snake.direction == Direction.DOWN

        state = [
            (dir_r and self.is_collision(Direction.RIGHT)) or # Danger straight
            (dir_l and self.is_collision(Direction.LEFT)) or
            (dir_u and self.is_collision(Direction.UP)) or
            (dir_d and self.is_collision(Direction.DOWN)),

            (dir_u and self.is_collision(Direction.RIGHT)) or # Danger right
            (dir_d and self.is_collision(Direction.LEFT)) or
            (dir_l and self.is_collision(Direction.UP)) or
            (dir_r and self.is_collision(Direction.DOWN)),

            (dir_d and self.is_collision(Direction.RIGHT)) or # Danger left
            (dir_u and self.is_collision(Direction.LEFT)) or
            (dir_r and self.is_collision(Direction.UP)) or
            (dir_l and self.is_collision(Direction.DOWN)),

            dir_l, #direction
            dir_r,
            dir_u,
            dir_d,

            self.main_game.fruit.pos.x < self.main_game.snake.body[0].x,  # fruit.pos left
            self.main_game.fruit.pos.x > self.main_game.snake.body[0].x,  # fruit.pos right
            self.main_game.fruit.pos.y < self.main_game.snake.body[0].y,  # fruit.pos up
            self.main_game.fruit.pos.y > self.main_game.snake.body[0].y  # fruit.pos down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
cell_size = 30
cell_number = 15

def update_training_plot(plot_scores, plot_mean_scores):
    # Ensure interactive mode is on
    plt.ion()

    # Prevent plot window from stealing focus (Windows only)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOACTIVATE = 0x0010
        HWND_TOP = 0
        mgr = plt.get_current_fig_manager()
        hwnd = mgr.window.winfo_id()
        user32.SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    except Exception as e:
        print("Could not prevent focus:", e)

    # Clear and update the plot
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(plot_scores)
    plt.plot(plot_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
    plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
    plt.pause(0.1)

def train():
    plot_scores = []
    plot_mean_scores = []
    
    total_score = 0
    record = 0
    pygame.init()
    
    agent = Agent()
    
    while True:
        # get old state
        state_old = agent.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = agent.main_game.move(final_move)
        state_new = agent.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.main_game.reset_game()    
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            update_training_plot(plot_scores, plot_mean_scores)
            


if __name__ == '__main__':
    train()


score = 0


#main_game = MAIN()




