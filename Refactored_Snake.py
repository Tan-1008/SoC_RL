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
from enum import Enum
import numpy as np

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
