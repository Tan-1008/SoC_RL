import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000
"""
env.observation_space. high and low are the highest and lowest possible values of state. In mountain-car, 
state is a tuple with 2 elements, position and velocity. And action space.n is the total number of actions,
here the total number of actions is 3 ie 0,1,2.
"""
print(env.observation_space.high)  
print(env.observation_space.low)
print(env.action_space.n)
"""
DISCRETE_OS_SIZE is going to be a 20x20 array and will contain all the possible combinations of discrete
position and velocity like we are making 20 discrete boxes and all continuous states will be put into
these 20 boxes for both position and velocity.
discrete_os_win_size is the window size of each discrete "box" so basically its going to be again a 2 elem
tuple which consists the size of position discrete state and velocity discrete state.
"""
DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print (discrete_os_win_size)
"""
this is creating the q table. A q table stores all the q-values for each action and each discrete state.
Q-value is sort of like return calculated for each action using a Bellmann formula. 
np.random.uniform spreads the values uniformly over the given low-high range. size here is [20,20] + [3]
so its basically a 20x20x3 table, where we have a q value mapping from every state possible to each of 
the 3 actions.

"""
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))
#env.reset() gives state, info where info isnt needed for basic implementations and it resets the env
#to original.
state, _ = env.reset()
for episode in EPISODES:
    discrete_state = get_discrete_state(state)
    #print (discrete_state) 

    done = False

    while not done:
        """
        COOL THING : to access like the argmax of velocity for  agiven action and position, you use
        np.argmax(q_table[5,:,2]) this basically will give the max velocity for 5th discrete position and 2.
        """
        action = np.argmax(q_table[discrete_state]) #picking the action for given discrete state which has max q value
        """
        action is executed, terminated - if goal reached or task failed(if it falls over in cartpole)
                            truncated  - if the episode hits max number of steps (usually 200)
        both bools
        """
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        env.render()            #opens pygame shows visual representation of environment
        done = terminated or truncated
        """
        this is the main loop which implements the formula for calcing new Q value.
        max future q is q(s',a') where a' is the action for which q(s',a') will have max value. 
        current_q is the current q value for that action which has just been implemented (need not be same as
        the action which will give next max future Q).
        NOTE : The (action,) is crucial since it creates a tuple with 1 element. If it was just (action) it
        wud just be stored as an integer and cant be concatenated to the tuple discrete_state.
        """
        if not done : 
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state+(action,)]= new_q
        
        elif new_state[0]>=env.goal_position : 
            q_table[discrete_state + (action,)]= 0
        """
        now this discrete state is used to decide next action in first line of while loop.
        """
        discrete_state = new_discrete_state
        

env.close()
