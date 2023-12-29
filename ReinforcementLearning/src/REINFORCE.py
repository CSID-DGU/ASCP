import gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.distributions import Categorical
from embedData import embedFlightData, flatten, print_xlsx, readXlsx
from functions import *
from CrewPairingEnv import CrewPairingEnv
import random

#Hyperparameters
gamma   = 0.98
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, N_flight, learning_rate):
        super(Policy, self).__init__()
        self.data = []

        self.N_flight = N_flight
        self.to(device)
        print("N_flight: ", self.N_flight)

        # 신경망 레이어 정의
        self.fc1 = nn.Linear(self.N_flight, 64)
        self.fc3 = nn.Linear(64, self.N_flight)
        
        torch.nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        # 옵티마이저 정의
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        x = flatten(x, self.N_flight)
        x = torch.tensor(x, dtype=torch.float32).to(device) 
        
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = torch.log(prob) * R
            
            loss.backward()
        
        self.optimizer.step()
        #self.scheduler.step()
        self.data = []

def main():
    current_directory = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(current_directory, '../dataset'))
    readXlsx(path, '/input_500.xlsx')

    flight_list, V_f_list = embedFlightData(path)
    
    # Crew Pairing Environment 불러오기
    N_flight = len(flight_list)
    env = CrewPairingEnv(V_f_list)
    pi = Policy(N_flight=N_flight, learning_rate=0.0002)
    pi.to(device)
    score = 0
    #scores = []
    bestScore= 99999999999999
    output = [[] for i in range(N_flight)]

    with open('episode_rewards.txt', 'w') as file:
        file.write("Episode\tReward\tBest Score\n")
        file.write("---------------------------------\n")
    
        for n_epi in range(500):
            print("############################ n_epi: ", n_epi, " ############################")
            s, _ = env.reset()  #현재 플라이트 V_P_list  <- V_f list[0]
            done = False
            output_tmp = [[] for i in range(N_flight)]
            
            while not done:            
                index_list = deflect_hard(env.V_p_list, s)
                prob = pi(index_list)
                
                selected_prob = prob[index_list]
                a = index_list[selected_prob.argmax().item()]
                
                s_prime, r, done, truncated, info = env.step(action=a, V_f=s)
                        
                pi.put_data((r,prob[a]))
                s = s_prime     #action에 의해 바뀐 flight
                score += r
                
                output_tmp[a].append(flight_list[env.flight_cnt-1].id)
                
            pi.train_net()
            if bestScore>score:
                bestScore=score
                output = output_tmp
            
            file.write(f"{n_epi}\t{score:.2f}\t{bestScore:.2f}\n")
            print(f"current score : {score:.2f} best score : {bestScore:.2f}")
            score=0
    
    env.close()
    
    print_xlsx(output)
    
if __name__ == '__main__':
    main()