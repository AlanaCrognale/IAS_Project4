# Heejin Chloe Jeong

import numpy as np

def get_action_egreedy(values ,epsilon):
	if np.random.rand() < epsilon: #perform randomly selected action
		a = np.random.choice(3,)
	else: #perform greedy action
		a = np.argmax(values) #choose action from state using policy derived from Q
	return a

def discretize(s):
    if s[0] < -0.3:
        s_0 = 0
    else:
        s_0 = 1

    if s[1] < 0:
        s_1 = 0
    else:
        s_1 = 1

    state = (s_0*(2**0))+(s_1*(2**1)) #like binary
    return state

def evaluation_cts_mountaincar(env, Q_table, step_bound = 1000, num_itr = 10):
	"""
	Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

	Input:
		env : an environment object.
		Q : A numpy array. Q values for all state and action pairs.
			Q.shape = (the number of states, the number of actions)
		step_bound : the maximum number of steps for each iteration
		num_itr : the number of iterations

	Output:
		Total number of steps taken to finish an episode (averaged over num_itr trials)
		Cumulative reward in an episode (averaged over num_itr trials)

	"""
	total_step = 0
	total_reward = -1
	itr = 0
	while(itr<num_itr):
		step = 0
		np.random.seed()
		state = discretize(env.reset())
		reward = -1
		done = False
		while((not done) and (step < step_bound)):
			action = get_action_egreedy(Q_table[state,:], 0.05)
			result = env.step(action)
			state_n = discretize(result[0])
			r = result[1]
			done = result[2]
			state = state_n
			reward += r
			step +=1
		total_reward += reward
		total_step += step
		itr += 1
	return total_step/float(num_itr), total_reward/float(num_itr)
