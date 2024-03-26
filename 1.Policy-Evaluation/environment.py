import numpy as np

HEIGHT = 5
WIDTH = 5

class grid_world:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT


    def is_terminal(self, state):   # Goal state
        x, y = state
        return False


    def interaction(self, state, action): #state is a list x, y
        if self.is_terminal(state):
            return state, 0

        def outOfBounds(state, action):
            if (state[0] + action[0] == -1 or state[0] + action[0] == 5 or state[1] + action[1] == -1 or state[1] + action[1] == 5):
                return True
            else: 
                return False

        #calculate rward and next state
        next_state = [0, 0]
        if state[0] == 0 and state[1] == 1: 
            reward = 10
            next_state[0] = 4
            next_state[1] = 1
        elif state[0] == 0 and state[1] == 3:
            reward = 5
            next_state[0] = 2
            next_state[1] = 3
        elif outOfBounds(state,action):
            reward = -1
            next_state = state
        else:
            reward = 0
            next_state[0] = state[0] + action[0]
            next_state[1] = state[1] + action[1]
        
        return next_state, reward


    def size(self):
        return self.width, self.height

env = grid_world()
ACTIONS = {'LEFT':np.array([0, -1]), 'UP':np.array([-1, 0]), 'RIGHT':np.array([0, 1]), 'DOWM':np.array([1, 0])}
for i in range(HEIGHT):
    for j in range(WIDTH): 
        for action in ACTIONS:
            (next_i, next_j), reward = env.interaction([i,j], ACTIONS[action])
            print(f'at position {i}, {j}, the action {action} is taken and the new position is {next_i}, {next_j} with reward {reward}')
        print("\n")
