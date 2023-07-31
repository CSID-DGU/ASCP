import gym
import numpy as np
from gym import spaces

class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set, cost_threshold):
        super(CrewPairingEnv, self).__init__()
        
        self.initial_pairing_set=initial_pairing_set
        self.pairing_set = initial_pairing_set
        self.current_cost=self.calculate_cost(self.pairing_set)
        self.cost_threshold = cost_threshold

        # Define action and observation space
        # Assuming the observation is the cost of each flight in the pairing set
        # And each action represents moving a flight from one pairing to another
        self.n_pairings=len(initial_pairing_set)
        self.max_flights = 5 #일단 임시로 5로 설정해둠.
        self.action_space = spaces.MultiDiscrete([self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(self.n_pairings, self.max_flights), dtype=np.int16)


    def step(self, action):

        source_pairing_index, flight_index, target_pairing_index, target_position = action

        # source pairing의 flight를, target pairing의 target position으로 이동시킴
        flight_to_move = self.pairing_set[source_pairing_index].pop(flight_index)
        self.pairing_set[target_pairing_index].insert(target_position, flight_to_move)

        # Calculate cost of the new pairing set
        new_cost = self.calculate_cost(self.pairing_set)

        # Check if the new cost meets the termination criteria
        done = new_cost <= self.cost_threshold

        # Calculate reward based on the improvement of the cost
        reward = self.current_cost - new_cost
        self.current_cost = new_cost

        # Return new state, reward, done, and additional info
        return self.pairing_set, reward, done, {}

    def calculate_cost(self, pairing_set):
        """
        Calculate the cost of the given pairing set.
        This should be implemented according to the specifics of the problem.
        """
        # Implement this method
        return 0

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.pairing_set = self.initial_pairing_set
        self.current_cost=self.calculate_cost(self.pairing_set)
        return self.pairing_set, self.current_cost