from turtle import forward
from types import new_class
import numpy as np
import random
from sympy import DiagonalOf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Linear_DQN(nn.Module):

    """ Linear_DQN: Use state embedding to make move decision. The model output is the Q-value of different actions given current state."""
    def __init__(self, input_size, hidden_size, output_size, Training = True):
        super(Linear_DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if not Training:
            self.model = self.load_state_dict(torch.load("training"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, filename = "training"):
        torch.save(self.state_dict(), filename)

class CNN_DQN(nn.Module):

    def __init__(self, input_channel, first_layer, kernel_size, Training = True):
        super(CNN_DQN, self).__init__()
        self.input_channel = input_channel
        self.first_layer = first_layer
        self.kernel_size = kernel_size

        # First layer 
        self.conv1 = nn.Conv2d(input_channel, self.first_layer, self.kernel_size)
        self.pool1 = nn.MaxPool2d(2,2)

        # Second layer 
        #self.conv2 = nn.Conv2d(first_layer, second_layer, self.kernel_size)
        #self.pool2 = nn.MaxPool2d(2,2)

        # Third layer 
        #self.conv3 = nn.Conv2d(second_layer, third_layer, self.kernel_size)
        #self.pool3 = nn.MaxPool2d(2,2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*4*5, 200)
        self.fc2 = nn.Linear(200, 3)

    def forward(self, x):
        # First layer
        # (N, 1, 60, 68)
        if len(x.shape) < 4:
            x = x[None,:]
        x = F.relu(self.conv1(x)) 
        # (N, 64, 56, 64)
        x = self.pool1(x)
        # (N, 64, 28, 32)

        # Second layer 
        x = F.relu(self.conv2(x))
        # (N, 128, 24, 28)
        x = self.pool2(x)
        # (N, 128, 12, 14)

        # Third layer
        x = F.relu(self.conv3(x))
        # (N, 256, 8, 10)
        x = self.pool3(x)
        # (N, 256, 4, 5)

        # FCC
        # x = self.flatten(x)
        # (N, 256*4*5)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, filename = "training_cnn"):
        torch.save(self.state_dict(), filename)

class Player:

    def __init__(self, model, alpha = 0.001, gamma = 0.9, epsilon = 1, batch_size = 100):
        self.num_games = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = []
        self.model = model 
        self.trainer = DQN_Trainer(self.model, self.alpha, self.gamma, self.batch_size)
        
    def get_embedded_state(self, game):

        """
            Three important information to make decisions:
            1. Danger near the snake
            2. Snake direction
            3. Food location
        """
        snake_x, snake_y = game.snake_x, game.snake_y

        danger_state = [snake_x > game.dis_width - 3*game.block,
                        snake_x < 3*game.block,
                        snake_y > game.dis_height - 3*game.block,
                        snake_y < 3*game.block]

        # Original direction 
        direction_state = [game.direction == 'left',
                           game.direction == 'right',
                           game.direction == 'up',
                           game.direction == 'down']

        # Food location
        food_state = [game.food_x < snake_x,
                      game.food_x > snake_x,
                      game.food_y < snake_y,
                      game.food_y > snake_y]

        state = danger_state + direction_state + food_state
        return np.array(state, dtype = int)

    def make_action_decision(self, game, state):
        # given current state, make action 
        if np.random.random() < self.epsilon:
            return game.action_space[game.direction][np.random.randint(3)]
        else:
            state = torch.tensor(state, dtype = torch.float)
            RL_action = torch.argmax(self.model(state)).item() # return the Q-function given the current state
        return game.action_space[game.direction][RL_action] # choose the action that maximise the Q-function

    def store_memory(self, state, action, reward, next_state, terminate):
        self.memory.append((state, action, reward, next_state, terminate))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            batch_sample = self.memory
        else:
            batch_sample = random.sample(self.memory, self.batch_size)
        
        # unpack the batch sample
        state, action, reward, next_state, terminate = zip(*batch_sample)
        self.trainer.train(state, action, reward, next_state, terminate)

    def train_short_memory(self, state, action, reward, next_state, terminate):
        """ Input: 
            -----------------------------------
            state: list with 12 dimension
            action: int 
            reward: int 
            next_state: list with 12 dimension
            terminate: bool
        """
        state = torch.tensor(state, dtype = torch.float) # torch.Size([12], float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float) # tensor float
        if terminate:
            target = reward
        else:
            target = reward + self.gamma*torch.max(self.model(next_state))
        label = self.model(state)
        TD_target = label.clone()
        TD_target[action] = target

        self.trainer.optimizer.zero_grad()
        loss = self.trainer.criterion(label, TD_target)
        loss.backward()
        self.trainer.optimizer.step()

class DQN_Trainer:

    def __init__(self, model, alpha, gamma, batch_size):
        self.alpha, self.gamma = alpha, gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.alpha)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def train(self, state, action, reward, next_state, terminate):
        #state = torch.stack(state)
        #next_state = torch.stack(next_state)
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)
        
        # Compte the TD Target 
        Q_val = self.model(state)
        TD_target = Q_val.clone()
        # if terminate is one-dimesnional

        for INDEX, DONE in enumerate(terminate):
            if DONE:
                Q_UPDATE = reward[INDEX]
            else:
                Q_UPDATE = reward[INDEX] + self.gamma*torch.max(self.model(next_state))

            TD_target[INDEX, action] = Q_UPDATE # we only do this at the point we take action is because the 'next state' is obtained through taking the action

        # Standard steps
        loss = self.criterion(TD_target, Q_val)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
    


    

