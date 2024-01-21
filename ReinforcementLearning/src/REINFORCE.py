import gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.distributions import Categorical
from embedData import embedFlightData, flatten, print_xlsx, readXlsx, unflatten
from functions import *
from CrewPairingEnv import CrewPairingEnv
from DK_Algorithm import *
import random

#Hyperparameters
gamma   = 0.98
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, N_flight, NN_size, learning_rate):
        super(Policy, self).__init__()
        self.data = []

        self.N_flight = N_flight
        self.NN_size = NN_size
        self.to(device)
        print("N_flight: ", self.N_flight)

        # 신경망 레이어 정의
        self.fc1 = nn.Linear(self.NN_size, 64)
        self.fc3 = nn.Linear(64, self.NN_size)
        
        torch.nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        # 옵티마이저 정의
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        x = flatten(x)
        print("flatten : ", x)
        x = torch.tensor(x, dtype=torch.float32).to(device) 
        
        x = F.leaky_relu(self.fc1(x))
        x = F.softmax(self.fc3(x), dim=0)

        print("prob : ", x)
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
    readXlsx(path, '/ASCP_Data_Input_873.xlsx')

    flight_list, V_f_list, NN_size = embedFlightData(path)
    
    # Crew Pairing Environment 불러오기
    N_flight = len(flight_list)
    env = CrewPairingEnv(V_f_list)
    pi = Policy(N_flight=N_flight, NN_size=NN_size, learning_rate=0.0002)

    # 저장한 모델 불러오기
    #load_model(pi, 'saved_model')

    pi.to(device)
    score = 0
    #scores = []
    bestScore= 99999999999999
    output = [[] for i in range(N_flight)]

    with open('episode_rewards.txt', 'w') as file:
        file.write("Episode\tReward\tBest Score\n")
        file.write("---------------------------------\n")
    
        for n_epi in range(1000):
            print("############################ n_epi: ", n_epi, " ############################")
            s, _ = env.reset()  #현재 플라이트 V_P_list  <- V_f list[0]
            done = False
            output_tmp = [[] for i in range(N_flight)]
            
            while not done:
                print("V_f : ", s)
                prob = pi(s)
                
                """
                index_list = deflect_hard(env.V_p_list, s)
                prob = pi(s)

                selected_prob = prob[index_list]
                a = index_list[selected_prob.argmax().item()]

                액션 선택 코드 다시 짜야함
                """

                good_pairing = unflatten(prob, NN_size)
                print("good : ", good_pairing)

                # good pairing이 현재 flight에 붙을 수 있는 지 검사
                checkConnection(good_pairing,s) # boolean으로 가능한 지 여부가 나옴
                
                # 유사도 검사를 통한 인덱스 반환
                a = find_similar(good_pairing, Pairing_list = env.V_p_list, flight = s)
                print(a)
                
                s_prime, r, done, truncated, info = env.step(action=a, V_f=s)
                        
                pi.put_data((r,prob[a]))
                s = s_prime     #action에 의해 바뀐 flight
                score += r
                
                output_tmp[a].append(flight_list[env.flight_cnt-1].id)
                
            pi.train_net()
            if bestScore>score:
                bestScore=score
                output = output_tmp

                # best score를 갱신하였으면 모델 저장
                #torch.save(pi.state_dict(), "saved_model")
            
            file.write(f"{n_epi}\t{score:.2f}\t{bestScore:.2f}\n")
            print(f"current score : {score:.2f} best score : {bestScore:.2f}")
            score=0
    
    env.close()
    
    print_xlsx(output)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    
if __name__ == '__main__':
    main()