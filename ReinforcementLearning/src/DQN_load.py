import os
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.distributions import Categorical
from embedData import *
from functions import *
from CrewPairingEnv import CrewPairingEnv
import random
import openpyxl
from datetime import datetime

#Hyperparameters
learning_rate = 0.005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, NN_Size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(NN_Size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    current_directory = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(current_directory, '../dataset'))
    readXlsx(path, '/input_873.xlsx')

    flight_list, V_f_list, NN_size = embedFlightData(path)

    # Crew Pairing Environment 불러오기
    N_flight = len(flight_list)
    env = CrewPairingEnv(V_f_list, flight_list)
    q = Qnet(NN_size)
    loaded_model = torch.load('dqn_model.pth')
    q.load_state_dict(loaded_model)
    q.eval()

    with open('episode_rewards.txt', 'w') as file:
        file.write("Episode\tReward\n")
        file.write("---------------------------------\n")
        time = datetime.now()

        score = 0
        output = []

        s, _ = env.reset()  #V_p 출발공항, V_f 도착공항
        done = False
        
        while not done:
            a = q.forward(s).argmax().item()
            # a = random.randint(0,1) # rule-based 코드

            s_prime, r, done, truncated, info, output = env.step(action=a)

            s = s_prime     #action에 의해 바뀐 flight
            score += r
        
        file.write(f"{score:.2f}\t{datetime.now()-time}\n")
        print(f"score : {score:.2f}")

    env.close()
    print_xlsx(output)
    
if __name__ == '__main__':
    main()