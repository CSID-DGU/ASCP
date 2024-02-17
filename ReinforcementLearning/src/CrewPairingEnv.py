## 라이브러리 임포트
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from embedData import embedFlightData
from functions import *

"""
여기서 gym.Env[np.ndarray, Union[int, np.ndarray]]는 gym.Env를 상속하면서 상태와 액션의 데이터 타입을 명시한 것입니다.

np.ndarray: 상태 공간의 데이터 타입으로 NumPy 배열을 의미합니다. -> V_shape의 shape를 가진 2차원 배열
Union[int, np.ndarray]: 액션 공간의 데이터 타입으로 정수 또는 NumPy 배열을 의미합니다. 
-> N_flight의 shape를 가진 1차원 배열 또는 정수
"""

class CrewPairingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description
    This code defines a Gym environment called CrewPairingEnv for a reinforcement learning task. 
    The environment is designed to solve a crew pairing optimization problem. In the airline industry, 
    crew pairing involves scheduling a set of flights for a group of crew members while satisfying various constraints and objectives.
    
    ### Action Space
    The action is a `ndarray` with shape `(flight_n,)` which can take values `{0, 1, ..., n_flight}` 
    indicating the available place where the current flight can be placed.

    | Num | Action                 |
    |-----|------------------------|
    | n_flight  | Available Pairing |

    **Note**: 액션을 통해서 하고자하는 일은 어떤 paring에 현재의 flight를 넣을 것인지를 결정하는 것임.
    따라서, 가능한 paring들을 모두 action space로 설정해줘야 함. (즉, 0~n_flight까지의 값이 action space가 됨)

    ### Observation Space(State)
    # 현재 배치해야하는 flight의 정보를 vector로 표현한 값.
    | Num | Observation           | Type                | Example              |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Origin Time         | int                |                |
    | 1   | Dest Time        | int                |                |
    | 2   | Duration            | int               | 480            |
    | 3   | Origin Airport      | one-hot vector      | LAX : [0, 0, 0, 0, 1] |
    | 4   | Dest Airport        | one-hot vector      | INC : [0, 1, 0, 0, 0] |    

    **Note:** 

    ### Rewards
    Since the goal is to reduce the cost, calculate the cost of the current pairing set and give a reward based on the cost.
    Give the lower reward as the cost is higher.
    Given the cost of the current pairing set, the reward is calculated as follows:
    reward = -cost
    
    ### Starting State
    Starting State는 제일 처음 들어오는 flight의 정보를 vector로 표현한 값으로 고정이 됨
    
    ### Episode End
    The episode ends if all the flights are placed in the pairing set.

    ### Arguments
    ```
    gym.make('CrewPairingEnv-ver1.1')
    ```

    No additional arguments are currently supported.
    """

    def __init__(self, V_f_list, flight_list):
        # number of flights
        self.V_f_list = V_f_list
        self.flight_list = flight_list
        self.N_flight = len(V_f_list)
        self.flight_cnt = 0
        self.V_p_cnt = 0
        self.V_p_list = [[0,0,0,[0],[0],[0]] for i in range(self.N_flight)]
        self.output = [[] for i in range(self.N_flight)]
        self.state = self.V_f_list[0]
        self.terminated = False
        
        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None # step() 함수가 호출되었을 때, terminated가 True인 경우를 의미함.
    

    def step(self, action):
        V_f = self.V_f_list[self.flight_cnt]
        if action == 1 :
            #print("flight cnt : ", self.flight_cnt)
            #print("V_p : ", self.V_p_list[self.V_p_cnt])
            #print("V_f : ", V_f)
            reward = get_reward(self.V_p_list, V_f, self.V_p_cnt)
            update_state(self.V_p_list, V_f, self.V_p_cnt)
            self.output[self.V_p_cnt].append(self.flight_list[self.flight_cnt].id)

            self.flight_cnt += 1
            self.V_p_cnt = 0

            if self.flight_cnt == self.N_flight :
                self.terminated = True
                return self.state, reward, self.terminated, False, {}, self.output
        else : 
            reward = 0
            self.V_p_cnt += 1

        V_f = self.auto_insert(self.V_f_list[self.flight_cnt])
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4]

        if self.flight_cnt == self.N_flight :
            self.terminated = True
        
        return self.state, reward, self.terminated, False, {}, self.output


    def reset(self):
        # number of flights
        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None # step() 함수가 호출되었을 때, terminated가 True인 경우를 의미함.
 
        self.steps_beyond_terminated = None
        self.V_p_list = [[0,0,0,[0],[0],[0]] for i in range(self.N_flight)]
        self.output = [[] for i in range(self.N_flight)]
        self.flight_cnt = 0
        self.terminated = False

        V_f = self.auto_insert(self.V_f_list[0])
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4] # V_p 출발공항 + V_f 도착공항
        
        return self.state, {}
    

    def auto_insert(self, V_f) :
        while True :
            if self.V_p_list[self.V_p_cnt] == [0,0,0,[0],[0],[0]] :
                self.V_p_list[self.V_p_cnt] = V_f
                self.output[self.V_p_cnt].append(self.flight_list[self.flight_cnt].id)
                
                self.flight_cnt += 1
                if self.flight_cnt == self.N_flight : break
                V_f = self.V_f_list[self.flight_cnt]

                self.V_p_cnt = 0
            
            elif checkConnection(self.V_p_list[self.V_p_cnt], V_f) == False : # or len(self.output[self.V_p_cnt])>=2 :
                self.V_p_cnt += 1

            else : break
        return V_f