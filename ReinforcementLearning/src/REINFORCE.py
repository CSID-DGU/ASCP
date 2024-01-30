import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.distributions import Categorical
from embedData import embedFlightData, flatten, print_xlsx, readXlsx, print_xlsx_tmp
from functions import *
from CrewPairingEnv import CrewPairingEnv
from DK_Algorithm import *
import random
import openpyxl
from datetime import datetime

#Hyperparameters
gamma   = 0.98
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self, NN_size, learning_rate):
        super(Policy, self).__init__()
        self.data = []

        self.NN_size = NN_size
        self.to(device)

        # 신경망 레이어 정의
        self.fc1 = nn.Linear(self.NN_size, 32)
        self.fc3 = nn.Linear(32, 2)
        
        torch.nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        # 옵티마이저 정의
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, a, b):
        x = flatten(a, b)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc3(x), dim=0)

        #print("prob : ", x)
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
    readXlsx(path, '/input_50000.xlsx')

    flight_list, V_f_list, NN_size = embedFlightData(path)
    #print("size: ", NN_size)
    
    # Crew Pairing Environment 불러오기
    N_flight = len(flight_list)
    env = CrewPairingEnv(V_f_list)
    pi = Policy(NN_size=NN_size, learning_rate=0.005)
    print(N_flight)
    # 저장한 모델 불러오기
    #load_model(pi, 'saved_model')

    pi.to(device)
    score = 0
    #scores = []
    bestScore= float('inf')
    output = [[] for i in range(N_flight)]

    with open('episode_rewards.txt', 'w') as file:
        file.write("Episode\tReward\tBest Score\n")
        file.write("---------------------------------\n")
        time = datetime.now()
    
        for n_epi in range(1):
            print("########################## n_epi: ", n_epi, " ############################  ", datetime.now()-time)
            s, _ = env.reset()  #현재 플라이트 V_P_list  <- V_f list[0]
            done = False
            output_tmp = [[] for i in range(N_flight)]
            
            while not done:
                for idx in range(len(env.V_p_list)):
                    V_p=env.V_p_list[idx]
                    if checkConnection(V_p,s)==False:
                        # print(idx)
                        continue
                    prob = pi(V_p, s)

                    if V_p == [0,0,0,[0],[0],[0]] : a = 1
                    elif prob.argmax().item() == 1 : a = 1
                    else : a = 0

                    s_prime, r, done, truncated, info, flag = env.step(action=a, V_f=s, idx=idx)

                    pi.put_data((r,prob[a]))

                    if flag :
                        # print(asd)
                        s = s_prime     #action에 의해 바뀐 flight
                        score += r
                        output_tmp[idx].append(flight_list[env.flight_cnt-1].id)
                        break

                
            pi.train_net()
            if bestScore>score:
                bestScore=score
                output = output_tmp

                # best score를 갱신하였으면 모델 저장
                #torch.save(pi.state_dict(), "saved_model")
            
            file.write(f"{n_epi}\t{score:.2f}\t{bestScore:.2f}\t{datetime.now()-time}\n")
            print(f"current score : {score:.2f} best score : {bestScore:.2f}")
                
            score=0

    env.close()
    
    print_xlsx(output)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    
if __name__ == '__main__':
    main()