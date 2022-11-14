import numpy as np
import pygame
import random
import os
from PIL import Image, ImageGrab
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn

window_position = (500, 250)
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (window_position)
tensor_transform = transforms.Compose([transforms.PILToTensor()]) # Transform image to tensor
grey_transform = transforms.Grayscale()

def image_transform(img, Gray = True, size = (60, 68)):
    img = grey_transform(img)
    img = fn.resize(img, size = size)
    img = tensor_transform(img)
    return img

#-----------------------------
# Color 
#-----------------------------
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
brown = (204, 102, 0)

class Snake_Game:
    
    def __init__(self, dis_width = 600, dis_height = 400, speed = 40):
        
        self.dis_width , self.dis_height = dis_width, dis_height
        self.x_init, self.y_init = self.dis_width/2, self.dis_height/2
        self.game_close = False

        self.speed = speed
        self.block = 10 # length of each block
        self.num_image = 0

        # Relative to the snake head: {forward, left, right}
        self.action_space = {'left':{0:'left',1:'down',2:'up'},
                            'right':{0:'right',1:'up',2:'down'},
                            'up':{0:'up',1:'left',2:'right'},
                            'down':{0:'down',1:'right',2:'left'}}
        
        self.action_dict = {0:'left', 1:'right', 2:'up', 3:'down'}

        self.boundary = []
        for i in range(self.dis_height):
            self.boundary.append((0,i))
            self.boundary.append((self.dis_width-self.block, i))
        for i in range(self.dis_width):
            self.boundary.append((i,0))
            self.boundary.append((i,self.dis_height-self.block))

        # Snake
        self.cord_change = {'left':(-self.block,0), 'right': (self.block,0), 'up':(0,-self.block), 'down':(0,self.block)}
        
        # Words
        pygame.font.init()
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("helvetica", 20)

    def new_game(self):
        pygame.init()
        self.snake_x, self.snake_y = self.x_init, self.y_init
        self.score = 0
        self.game_over = False
        self.direction = 'right'
        self.snake = [(self.x_init, self.y_init)]

        self.dis = pygame.display.set_mode((self.dis_width, self.dis_height))   
        pygame.display.set_caption('Snake')
        self.dis.fill(blue)
        self.clock = pygame.time.Clock()
        self.gen_food()
        pygame.display.update()
        
    def one_shot_screen(self):
        self.snake_x += self.cord_change[self.direction][0]
        self.snake_y += self.cord_change[self.direction][1]

        self.snake.append((self.snake_x, self.snake_y))
        del self.snake[0]
            
        # Print the screen 
        self.dis.fill(blue)
        pygame.draw.rect(self.dis, green, [self.food_x, self.food_y, self.block, self.block])
        self.print_snake()
        self.print_boundary()
        pygame.display.update()
        
        if (self.snake_x, self.snake_y) in self.boundary:
            self.game_over = True
        if (self.snake_x, self.snake_y) in self.snake[:-1]:
            self.game_over = True

        if self.snake_x == self.food_x and self.snake_y == self.food_y:
            self.gen_food()
            pygame.image.save(self.dis, 'screenshot.jpg')
            self.snake.append((self.snake_x, self.snake_y))
            self.score += 1
            
        self.clock.tick(self.speed)
            
    def print_snake(self):
        for i, x in enumerate(self.snake):
            if i == len(self.snake) - 1:
                pygame.draw.rect(self.dis, red ,[x[0], x[1], self.block, self.block])
            else:
                pygame.draw.rect(self.dis, black ,[x[0], x[1], self.block, self.block])
        
    def print_boundary(self):
        for z in self.boundary:
            pygame.draw.rect(self.dis, brown, [z[0], z[1], self.block, self.block])
            
    def make_action(self, move):
        self.direction = move

    def gen_food(self):
        self.food_x = round(random.randrange(self.block, self.dis_width - 2*self.block) / 10.0) * 10.0
        self.food_y = round(random.randrange(self.block, self.dis_height - 2*self.block) / 10.0) * 10.0

    def message(self, msg, color):
        mesg = self.font_style.render(msg, True, color)
        self.dis.blit(mesg, [self.dis_width / 6, self.dis_height / 3])

    def get_screenshot(self, ToTensor = True, ToGrey = True):
        img = ImageGrab.grab(bbox = (window_position[0], window_position[1], window_position[0] + self.dis_width, window_position[1] + self.dis_height))
        return image_transform(img)

    def food_distance(self):
        return (self.snake_x - self.food_x)**2 + (self.snake_y - self.food_y)**2

    def game_over_screen(self):
        self.dis.fill(blue)
        self.message("You Lost! Press C-Play Again or Q-Quit", red)
        value = self.score_font.render("Your Score: " + str(self.score), True, yellow)
        self.dis.blit(value, [0, 0])
        pygame.display.update()


def run_new_game(Player = "HumanPlayer"):
    game = Snake_Game()
    game.new_game()

    while not game.game_close:
        while not game.game_over:
            if Player == "HumanPlayer":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.game_over = True
                        game.game_close = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            game.make_action('left')
                        elif event.key == pygame.K_RIGHT:
                            game.make_action('right')
                        elif event.key == pygame.K_UP:
                            game.make_action('up')
                        elif event.key == pygame.K_DOWN:
                            game.make_action('down')
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.game_over = True
                        game.game_close = True
                game.make_action(game.action_space[game.direction][np.random.randint(3)])
            game.one_shot_screen()

        game.game_over_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_close = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game.game_close = True
                if event.key == pygame.K_c:
                    run_new_game()
    pygame.quit()
    quit()
    
def machine_new_game():
    game = Snake_Game()
    game.new_game()

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.make_action('left')
                elif event.key == pygame.K_RIGHT:
                    game.make_action('right')
                elif event.key == pygame.K_UP:
                    game.make_action('up')
                elif event.key == pygame.K_DOWN:
                    game.make_action('down')
        game.one_shot_screen()
    pygame.quit()
    quit()

if __name__ == "__main__":
    run_new_game("RandomPlayer")
    #machine_new_game()