'''
###############################################################################################
	Black Jack (Reinforcement Learning: An Introduction, Sutton, Barto)

	Program is about the Black-Jack in Monte-Carlo chapter, presented
	in the Reinforcement Learning: An Introduction book, Sutton, Barto, 
###############################################################################################
'''

import numpy as np
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




class Agent(object):
	def __init__(self):
		self.value_estimates=np.zeros((10,10))
		self.sample_returns=np.zeros((10,10))
		self.episode_map=[]
		self.dealer_num=None
		self.num_count=np.zeros((10,10))
		self.initial_state=np.random.randint(low=11,high=21)
		self.final_player_sum=None

	def action(self,player_sum):
		response=None
		if player_sum<20:
			response="hit"
		if player_sum==20 or player_sum==21:
			response="stick"

		return response

	def episode(self,dealer_num,player_sum):
		state=(dealer_num,player_sum)
		self.episode_map.append(state)
		self.num_count[dealer_num-1][player_sum-12]+=1

	def first_visit_mc(self,reward):
		index=None
		for i in range(len(self.episode_map)):
			index=self.episode_map[i]
			dealer_num=index[0]-1
			player_sum=index[1]-12
			self.value_estimates[dealer_num][player_sum]+=reward   

	def reset_episode(self):
		self.episode_map*=0

	def reset(self):
		self.value_estimates*=0
		self.num_count*=0

	def print(self):
		print("value_estimates:" + str(self.value_estimates))
		print("counts: "+ str(self.num_count))



class Environment(object):
	def __init__(self, agents, num_episodes):

		self.agents = agents
		self.agents.reset()
		self.num_episodes = num_episodes


	def dealer_response(self,dealer_num):
		while dealer_num<17:
			dealer_num+=np.random.randint(low=1,high=11)

		return dealer_num


	def reward_function(self,dealer_num,player_sum):
		reward=None
		if player_sum<=21 and player_sum>dealer_num:
			reward=1
		if player_sum<=21 and player_sum==dealer_num:
			reward=0
		if player_sum<=21 and dealer_num<=21 and player_sum<dealer_num:
			reward=1
		if player_sum>21:
			reward=-1
		if player_sum<=21 and dealer_num>21:
			reward=1

		return reward


	def play(self):
		for num_episodes in tqdm(range(self.num_episodes)):
			#print("New episode started")
			#self.agents.print()


			#generate number for dealer
			self.agents.reset_episode()
			dealer_num=np.random.randint(low=1,high=11)
			player_sum=np.random.randint(low=12,high=22)
			initial_state=self.agents.initial_state
			#print("Dealer Value: "+ str(dealer_num))
			#print("Player sum: " + str(player_sum))
			self.agents.episode(dealer_num,player_sum)  ##only beginning state  is added to episode
			

			while player_sum<=21:
				#self.agents.episode(dealer_num,player_sum)  ##all the states are added to episode
				response=self.agents.action(player_sum)
				#print("Response generated: " + str(response))

				if response=="hit":
					player_sum+=np.random.randint(low=1,high=11)
					#print("hit action called ")
					#print("new player sum: " + str(player_sum))

				if response=="stick":
					#self.agents.episode(dealer_num,player_sum)
					self.agents.final_player_sum=player_sum
					#print("stick called thus break will be called")
					break

			if player_sum>21:
				self.agents.final_player_sum=player_sum

			final_dealer_sum=self.dealer_response(dealer_num)
			reward=self.reward_function(final_dealer_sum,self.agents.final_player_sum)
			self.agents.first_visit_mc(reward)
			#print("Epsiode is :" +str(self.agents.episode_map))

		self.agents.value_estimates=self.agents.value_estimates/self.agents.num_count
		return self.agents.value_estimates

################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution                 
	num_episodes= 500000                  # number of episodes

	agents = Agent()
	environment = Environment(agents=agents,num_episodes=num_episodes)

	# Run Environment
	print("Running...")
	V = environment.play()
	print(V)
	print("Execution time: %s seconds" % (time.time() - start_time))
	
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')

	for index_x in range(10):
		for index_y in range(10):
			ax.scatter(index_x+1, index_y+12, V[index_x][index_y],c='r', marker='o')
	
	#Graph1
	ax.set_title('BLACK JACK STATE ESTIMATION')
	ax.set_xlabel('Dealer Initial Value')
	ax.set_ylabel('Player Sum')
	ax.set_zlabel('Average Reward')

	plt.show()








