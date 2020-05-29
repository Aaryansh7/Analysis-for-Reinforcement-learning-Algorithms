'''
###############################################################################################
	Black Jack with Exploring Start (Reinforcement Learning: An Introduction, Sutton, Barto)

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
		#self.action_value_estimates=np.zeros((10,10))
		#self.sample_returns=np.zeros((10,10))
		self.episode_map=[]
		self.dealer_num=None
		#self.num_count=np.zeros((10,10))
		#self.initial_state=np.random.randint(low=11,high=21)
		#self.final_player_sum=None
		self.main_list=[]
		self.key=-1
		self.key_link=-1
		self.policy=np.zeros(400)

	def add_state(self,dealer_num,player_sum,ace,action):

		for i in range(len(self.main_list)):
			dict=self.main_list[i]
			if (dict["Ace"]==ace and dict["Player_sum"]==player_sum and dict["Dealer_sum"]==dealer_num and dict["Action"]==action):
				dict["Count"]+=1
				return dict["key"]

		self.key+=1
		key_link=None
		for i in range(len(self.main_list)):
			dict=self.main_list[i]
			if (dict["Ace"]==ace and dict["Player_sum"]==player_sum and dict["Dealer_sum"]==dealer_num and dict["Action"]!=action):
				key_link=dict["key-link"]
				break

		if key_link=None:
			self.key_link+=1

		dict={	
				"key":self.key
				"key-link":self.key_link
				"Ace":ace,
				"Player_sum":player_sum,
				"Dealer_sum":dealer_num,
				"Action":action
				"Action_Value":None
				"Count":1
				}

		self.main_list.append(dict)

		return dict["key"]

	def ace_state(self,player_sum):
		if player_sum+11<=21:
			ace_status=1 				#value:1 for usable ace											
		else:
			ace_status=0

		return ace_status

	def episode(self,key):
		self.episode_map.append(key)


	def compute_action_value(self,reward):

		for i in range(len(self.episode_map)):
			key=self.episode_map[i]
			for j in range(len(self.main_list)):
				dict=self.main_list[i]
				if dict["key"]==key:
					if dict["Count"]==1:
						dict["Action_Value"]=((dict["Action_Value"]*dict["Count"])+reward)/dict["Count"]
					else:
						dict["Action_Value"]=((dict["Action_Value"]*(dict["Count"]-1))+reward)/(dict["Count"]-1)


	def policy_update(self,episode_map):

		for i in range(len(episode_map)):
			key_link=None
			action=None
			key=episode_map[i]

			top=float("-inf")

			for j in range(len(self.main_list)):
				dict=self.main_list[j]
				if dict["key"]==key:
					key_link=dict["key-link"]
					if top<dict["Action_Value"]:
						action_val=(dict["Action_Value"])
						action=dict["Action"]
						top=action_val

			self.policy[key_link]=action



	def action(self,dealer_num,player_sum,ace):
		key_link=None

		for i in range(len(self.main_list)):
			dict=self.main_list[i]
			if (dict["Ace"]==ace and dict["Player_sum"]==player_sum and dict["Dealer_sum"]==dealer_num ):
				key_link=dict["key-link"]

		if self.policy[key_link]!=None:
			response=self.policy[key_link]

		if key_link==None:
			response=None
			if player_sum<20:
				response="hit"
			if player_sum==20 or player_sum==21:
				response="stick"

		return response


	def reset_episode(self):
		self.episode_map*=0


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

		ace_status_main=None
		start=0
		reesponse=None

		for num_episodes in tqdm(range(self.num_episodes)):
		
			self.agents.reset_episode()
			dealer_num=np.random.randint(low=1,high=11)
			player_sum=np.random.randint(low=12,high=22)

			random_action=np.random.randint(low=0,high=2)
			if random_action==0:
				response="hit"
			else
				response="stick"

			key=self.agents.add_state(dealer_num,player_sum,ace_status_main=0,response)
				self.agents.episode(key)

			while player_sum<=21:

				if response=="hit":
					ace_status=np.random.randint(low=0,high=2)

					if ace_status==0:
						player_sum+=np.random.randint(low=1,high=11)
						ace_status_main=0

					if ace_status==1:
						ace_status_main=self.agents.ace_status(player_sum)
						if ace_status_main==1:
							player_sum+=11
						else:
							player_sum+=1


				response=self.agents.action(dealer_num,player_sum,ace_status_main)

				key=self.agents.add_state(dealer_num,player_sum,ace_status_main,response)
				self.agents.episode(key)

				if response=="stick":
					self.agents.final_player_sum=player_sum
					break

			if player_sum>21:
				self.agents.final_player_sum=player_sum

			final_dealer_sum=self.dealer_response(dealer_num)
			reward=self.reward_function(final_dealer_sum,self.agents.final_player_sum)
			self.agents.compute_action_value(reward)
			self.agents.policy_update(episode_map)


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








