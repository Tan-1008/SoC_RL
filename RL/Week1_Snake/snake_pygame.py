import pygame
import sys, random
from pygame.math import Vector2
class FRUIT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x*cell_size),int(self.pos.y*cell_size),cell_size,cell_size)
        screen.blit(apple,fruit_rect)   
    def randomize(self):
        self.x = random.randint(0,cell_number-1)
        self.y = random.randint(0,cell_number-1)
        self.pos = pygame.math.Vector2(self.x,self.y)

class SNAKE:
    def __init__(self):
        self.direction=Vector2(1,0)
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.new_block = False

    def draw_snake(self):
        for block in self.body :
            snake_rect = pygame.Rect(int(block.x*cell_size),int(block.y*cell_size),cell_size,cell_size)
            pygame.draw.rect(screen,(pygame.Color("red")),snake_rect)
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
        self.snake = SNAKE()
        self.fruit = FRUIT()
    
    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()
    
    def draw_elements(self):
        self.fruit.draw_fruit()
        self.snake.draw_snake()

    def check_collision(self):
        if self.fruit.pos==self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()
    def check_fail(self):
        if self.snake.body[0].x not in range(0,cell_number) or self.snake.body[0].y not in range(0,cell_number):
            self.game_over()
        for block in self.snake.body[1:]:
            if self.snake.body[0] == block :
                self.game_over()

    def game_over(self):
        pygame.quit()
        sys.exit()



pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_size*cell_number,cell_size*cell_number))
clock = pygame.time.Clock()
apple = pygame.image.load("Graphics/apple.png").convert_alpha()


test_rect = pygame.Rect(100,200,100,100)
x_pos = 200
main_game = MAIN()

SCREEN_UPDATE = pygame.USEREVENT                #some event we can trigger
pygame.time.set_timer(SCREEN_UPDATE,150)   #we trigger it using this timer 

while True : 
    # draw all our elements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            main_game.game_over()
        if event.type == SCREEN_UPDATE:
            main_game.update()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if main_game.snake.direction.y != 1 :
                    main_game.snake.direction = Vector2(0,-1)
            if event.key == pygame.K_RIGHT:
                if main_game.snake.direction.x != -1 :
                    main_game.snake.direction = Vector2(1,0)
            if event.key == pygame.K_DOWN:
                if main_game.snake.direction.y != -1 :
                    main_game.snake.direction = Vector2(0,1)
            if event.key == pygame.K_LEFT:
                if main_game.snake.direction.x != 1 :
                    main_game.snake.direction = Vector2(-1,0)                
        
    screen.fill((175,215,70))
    main_game.draw_elements()
    
    
    pygame.display.update()
    clock.tick(60)
