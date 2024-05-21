import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]), np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            dp_results = np.load('./result/dp.npz')
            self.values = dp_results['V']
            self.policy = dp_results['PI']
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS) 




    def policy_evaluation(self, iter, env, policy, discount=1.0): #iter:int env:object policy:array(np) discount:float
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 0
        
        while (iteration < iter):
            for height in range(HEIGHT): 
                for width in range(WIDTH): #for every state x, y
                    value = 0 
                    for action_index in range(len(ACTIONS)): #weighted average of every action 
                        action = ACTIONS[action_index] 
                        new_state , reward =  env.interaction([height, width], action)
                        action_prob = policy[height, width, action_index]
                        value += action_prob * (reward + self.values[new_state[0], new_state[1]])
                    new_state_values[height, width] = value
            iteration = iteration + 1

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration





    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()
        policy_stable = True
        for height in range(HEIGHT):
            for width in range(WIDTH):
                old_action = np.argmax(policy[height, width])
                action_values = np.zeros(len(self.ACTIONS))

                for action_index in range(len(self.ACTIONS)):
                    action = ACTIONS[action_index]
                    new_state, reward = env.interaction([height, width], action)
                    action_values[action_index] = reward +  state_values[new_state[0], new_state[1]]

                best_action = np.argmax(action_values)
                policy[height, width] = np.eye(len(self.ACTIONS))[best_action]
                if old_action != best_action:
                    policy_stable = False
                    


        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

