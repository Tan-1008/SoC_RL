"""
BASIC SNAKE GAME USING PYGAME

Some additional features I added :
1. A background checkered board 
2. A score counter in the bottom right to keep track during the game too
3. A check to make sure fruit isnt spawned at any current block occupied by snake

"""
import pygame
import sys, random
from pygame.math import Vector2
class FRUIT:
    def __init__(self,snake_body):                  #using snake_body as an argument to use the self.body from snake class in fruit class
        self.randomize(snake_body)

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x*cell_size),int(self.pos.y*cell_size),cell_size,cell_size)
        pygame.draw.rect(screen,(126,166,114),fruit_rect)
    def randomize(self,snake_body):
        while True :
            self.x = random.randint(0,cell_number-1)
            self.y = random.randint(0,cell_number-1)
            new_pos = Vector2(self.x,self.y)
            if new_pos not in snake_body :          #imp feature imo that both resources missed, where fruit shouldnt spawn where snake's body is present
                self.pos = new_pos                  #this makes game less confusing when snake gets long enough
                break

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
        self.fruit = FRUIT(self.snake.body)
    
    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()
    
    def draw_elements(self):
        self.draw_grass()
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        self.show_score()
        
    def show_start_screen(self):
        title_font = pygame.font.SysFont('times new roman', 60, bold=True)
        button_font = pygame.font.SysFont('times new roman', 30)

        title_surface = title_font.render("Cool Snake Game", True, pygame.Color("green"))
        title_rect = title_surface.get_rect(center=(cell_size * cell_number / 2, cell_size * cell_number / 4))

        button_rect = pygame.Rect(0, 0, 200, 60)
        button_rect.center = (cell_size * cell_number / 2, cell_size * cell_number / 2)

        button_text = button_font.render("Start Game", True, (255, 255, 255))
        button_text_rect = button_text.get_rect(center=button_rect.center)

        while True:
            screen.fill((0, 0, 0))
            screen.blit(title_surface, title_rect)

            pygame.draw.rect(screen, (0, 128, 255), button_rect)
            screen.blit(button_text, button_text_rect)
            
            instructions = button_font.render("Use arrow keys to move the snake", True, (200, 200, 200))
            instructions_rect = instructions.get_rect(center=(cell_size * cell_number / 2, cell_size * cell_number * 3 / 4))
            screen.blit(instructions, instructions_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        return  # Exit start screen and begin game

    def show_score(self):
        score_font = pygame.font.SysFont('comic sans ms',25)
        self.score = str(len(self.snake.body)-3)
        score_surface =  score_font.render("Score : "+ self.score,True,pygame.Color("purple"))
        score_x = int(cell_size * cell_number - 60)
        score_y = int (cell_number*cell_size - 40)
        score_rect = score_surface.get_rect(center = (score_x,score_y))
        screen.blit(score_surface,score_rect)

    def draw_grass(self):
        grass_color = (167, 209, 60)
        for row in range(cell_number):
            if row%2 == 0 :
                for col in range(cell_number):
                    if col%2 == 0:
                        grass_rect = pygame.Rect(col*cell_size,row*cell_size,cell_size,cell_size)
                        pygame.draw.rect(screen,grass_color,grass_rect)
            else :
                for col in range(cell_number):
                    if col%2 != 0:
                        grass_rect = pygame.Rect(col*cell_size,row*cell_size,cell_size,cell_size)
                        pygame.draw.rect(screen,grass_color,grass_rect)    

    def check_collision(self):
        if self.fruit.pos==self.snake.body[0]:
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
    def check_fail(self):
        if self.snake.body[0].x not in range(0,cell_number) or self.snake.body[0].y not in range(0,cell_number):
            self.game_over()
        for block in self.snake.body[1:]:
            if self.snake.body[0] == block :
                self.game_over()

    def game_over(self):
        my_font = pygame.font.SysFont('times new roman', 50, bold=True)
        small_font = pygame.font.SysFont('times new roman', 30)

        game_over_surface = my_font.render('Your Score is : ' + str(self.score), True, pygame.Color("red"))
        game_over_rect = game_over_surface.get_rect(center=(cell_size * cell_number / 2, cell_size * cell_number / 3))
        
        
        button_width, button_height = 200, 60
        center_x = cell_size * cell_number / 2

        play_again_rect = pygame.Rect(0, 0, button_width, button_height)
        play_again_rect.center = (center_x, cell_size * cell_number / 2)

        quit_rect = pygame.Rect(0, 0, button_width, button_height)
        quit_rect.center = (center_x, cell_size * cell_number * 2 / 3)

        while True:
            screen.fill((0, 0, 0))  
            screen.blit(game_over_surface, game_over_rect)

            
            pygame.draw.rect(screen, (0, 200, 0), play_again_rect)  
            pygame.draw.rect(screen, (200, 0, 0), quit_rect)        

           
            play_text = small_font.render("Play Again", True, (255, 255, 255))
            quit_text = small_font.render("Quit", True, (255, 255, 255))

            play_text_rect = play_text.get_rect(center=play_again_rect.center)
            quit_text_rect = quit_text.get_rect(center=quit_rect.center)

            screen.blit(play_text, play_text_rect)
            screen.blit(quit_text, quit_text_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_again_rect.collidepoint(event.pos):
                        self.reset_game()  
                        return  

                    if quit_rect.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()
    def reset_game(self):
        self.snake = SNAKE()       # reset snake position, direction, and body
        self.fruit = FRUIT(self.snake.body) 


pygame.init()
cell_size = 40
cell_number = 20
pygame.display.set_caption('Cool Snake Game')
screen = pygame.display.set_mode((cell_size*cell_number,cell_size*cell_number))

clock = pygame.time.Clock()
score = 0


test_rect = pygame.Rect(100,200,100,100)
x_pos = 200
main_game = MAIN()
main_game.show_start_screen()

SCREEN_UPDATE = pygame.USEREVENT                
pygame.time.set_timer(SCREEN_UPDATE,150)   

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
        
    screen.fill((175,215,69))
    main_game.draw_elements()
    
    
    pygame.display.update()
    clock.tick(69)
