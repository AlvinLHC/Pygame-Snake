To test different models in table 1:
Change line 109 in main.py and line 19 in RL_Toolkit.py

the table file information:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TABLE NAME  MODEL 				   main.py line 109 						RL_Toolkit line 19
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
training    1 layer 256 neurons (baseline model)   model = Linear_DQN(12, 256, 3, Training=False) 		self.model = self.load_state_dict(torch.load("training"))
training2   1 layer 128 neurons 		   model = Linear_DQN(12, 128, 3, Training=False)  		self.model = self.load_state_dict(torch.load("training2"))
training3   2 layer (64, 128) neurons	           model = Linear_DQN(12, 64, 128, 3, Training=False) 		SEE BELOW
training4   1 layer 512 neurons		 	   model = Linear_DQN(12, 512, 3, Training=False) 		self.model = self.load_state_dict(torch.load("training4"))


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Replace line 14 to 24 in RL_Toolkit.py if you want to test the 2 layer (64, 128) neurons model:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def __init__(self, input_size, hidden_size1, hidden_size2, output_size, Training = True):
        super(Linear_DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        if not Training:
            self.model = self.load_state_dict(torch.load("training3"))

def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x