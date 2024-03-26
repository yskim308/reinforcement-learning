import numpy as np
from environment import grid_world
from visualize import draw_image

WORLD_SIZE = 5
# left, up, right, down
ACTIONS = {'LEFT':np.array([0, -1]), 'UP':np.array([-1, 0]), 'RIGHT':np.array([0, 1]), 'DOWM':np.array([1, 0])}
ACTION_PROB = 0.25



def evaluate_state_value_by_matrix_inversion(env, discount=0.9):
    WIDTH, HEIGHT = env.size()

    # Reward matrix R
    R = np.zeros((WIDTH, HEIGHT))
    for i in range(WIDTH):
        for j in range(HEIGHT):
            expected_reward = 0
            for action in ACTIONS:
                (next_i, next_j), reward = env.interaction([i, j], ACTIONS[action])
                expected_reward += ACTION_PROB*reward
            R[i, j] = expected_reward
    print(R)
    R = R.reshape((-1,1))

    # Transition matrix  
    P = np.zeros((WIDTH*HEIGHT, WIDTH*HEIGHT))
    for i in range(WIDTH):
        for j in range(HEIGHT):
            current_state = i*HEIGHT + j
            for action in ACTIONS: 
                (next_i, next_j), _ = env.interaction([i,j], ACTIONS[action])
                next_state = next_i*HEIGHT + next_j
                P[current_state, next_state] += ACTION_PROB
        
    print(P)
    #invert matrix to calculate V 
    V = np.linalg.inv(np.eye(WIDTH*HEIGHT) - discount*P).dot(R)         

    new_state_values = V.reshape(WIDTH,HEIGHT)
    draw_image(1, np.round(new_state_values, decimals=2))

    return new_state_values


if __name__ == '__main__':
    env = grid_world()
    values = evaluate_state_value_by_matrix_inversion(env = env)



