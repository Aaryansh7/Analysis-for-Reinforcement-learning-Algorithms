import gym
from agents import RandomAgent
from agents import ValueIterAgent
import numpy as np
env = gym.make('FrozenLake-v0')
gamma = 1

#Step1: Instantiate a Random/ValueIter Agent
#agent=RandomAgent(env.action_space)
agent=ValueIterAgent(env,gamma)
#Step2: For Value Iter Agent, Evaluate Policy
agent.value_iteration()
agent.extract_policy()

print("Agent Policy" + str(agent.policy))
#Step3: Play Frozen Lake 1000 times with this policy and measure rewards
all_reward=[]
for episode in range(1000):
	obs=env.reset()
	total_reward=0

	while True:
		action=agent.choose_action(obs)
		obs,reward,done,info=env.step(action)

		if done:
			all_reward.append(reward)
			break

print("Average Reward: " +str (np.mean(all_reward)))
#Step4: Print Average Reward	    	

   	

