import torch
import random
import numpy as np
from Snake import Snake_Game, run_new_game
from RL_Toolkit import DQN_Trainer, Player, Linear_DQN, CNN_DQN
import os 

clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')
clear()

def TRAIN_AI(method = "Linear", num_games = 600):

    """ Main code to train the computer to play the game."""
    
    if method == "Linear":
        input_channel, hidden_channel, output_channel = 12, 128, 3
        model = Linear_DQN(12, 32, 64, 128, 3)
    elif method == "CNN":
        pixel = (60, 68)
        input_channel = 1
        first_layer_dim = 64
        second_layer_dim = 128
        third_layer_dim = 256
        kernel_size = 5
        model = CNN_DQN(input_channel, first_layer_dim, second_layer_dim, third_layer_dim, kernel_size)

    scores_hist = []
    scores_mean_hist = []
    game = Snake_Game()
    agent = Player(model)
    record = 0
    step = 0
    maxStep = 600
    epsilon_decay = 0.005
    if method == "Linear":
        filename = "scores_linear_32_64_128.txt"
    elif method == "CNN":
        filename = "scores_cnn.txt"

    open(filename, "w").close()

    for i_games in range(num_games):
        game.new_game()
        while not game.game_over and step <= maxStep:
            if method == "Linear":
                old_state = agent.get_embedded_state(game)
            elif method == "CNN":
                old_state = game.get_screenshot()
            action_str = agent.make_action_decision(game, old_state) # str in {'left','right','up','down'}
            action_dict = game.action_space[game.direction]
            action_num = list(action_dict.keys())[list(action_dict.values()).index(action_str)] # int

            old_scores = game.score
            old_distance = game.food_distance()

            game.make_action(action_str) # void
            game.one_shot_screen() # void 
            step += 1
            game_over = game.game_over # bool
            scores = game.score
            new_distance = game.food_distance()

            if game.score > old_scores:
                reward = 50
                step = 0
            elif game_over:
                reward = -20
            else:
                reward = -0.01
                #if new_distance < old_distance:
                #    reward = 0.02
                #else:
                #    reward = -0.02 # Discourage long move 

            if method == "Linear":
                new_state = agent.get_embedded_state(game) # list with 12 dimension
            else:
                new_state = game.get_screenshot()

            if method == "Linear":
                agent.train_short_memory(old_state, action_num, reward, new_state, game_over)
            if method == "CNN":
                old_state2 = old_state[None,:]
                new_state2 = new_state[None,:]
                agent.train_short_memory(old_state2, action_num, reward, new_state2, game_over)
            agent.store_memory(old_state, action_num, reward, new_state, game_over)
            
        if scores > record:
            record = scores
            agent.model.save()
        
        step = 0
        scores_hist.append(scores)
        scores_mean_hist.append(np.mean(scores_hist))
        agent.train_long_memory()

        # Reduce the exploration probability 
        if i_games > 200:
            agent.epsilon -= epsilon_decay
        #--------------------------------------------------
        # PRINT RESULT
        #--------------------------------------------------

        with open(filename, "a") as f:
            f.write(str(scores) + "\n") 
        print(f'Game: {i_games+1}, scores: {scores}, record: {max(scores_hist)}.\n')

def TEST_AI(num_games = 50):
    model = Linear_DQN(12, 256, 3, Training=False)
    game = Snake_Game()
    agent = Player(model, epsilon=0)
    scores_hist = []
    maxStep = 400
    step = 0
    for i_games in range(num_games):
        game.new_game()
        while not game.game_over and step < maxStep:
            old_state = agent.get_embedded_state(game)
            action_str = agent.make_action_decision(game, old_state) # str in {'left','right','up','down'}
            action_dict = game.action_space[game.direction]
            action_num = list(action_dict.keys())[list(action_dict.values()).index(action_str)] # int

            old_scores = game.score
            game.make_action(action_str) # void
            game.one_shot_screen() # void 
            if game.score > old_scores:
                step = 0
            step += 1
            game_over = game.game_over # bool
            scores = game.score
        step = 0
        scores_hist.append(scores)
        #with open("testing_scores_512.txt",'a') as f:
        #    f.write(str(scores) + '\n')
        print(f'Game: {i_games+1}, scores: {scores}, record: {max(scores_hist)}.\n')
    print('-------------------------------------------')
    print(f'Game Summary: Average scores = {np.mean(scores_hist)}, max score = {max(scores_hist)}.')

if __name__ == "__main__":
    TRAIN_MODEL = False
    if TRAIN_MODEL:
        TRAIN_AI()
    else:
        TEST_AI()
        


        
