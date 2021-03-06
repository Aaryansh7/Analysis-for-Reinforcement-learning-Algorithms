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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors


def accomdate_axis_change(pi):
	##changing pi
	final_array=np.zeros((11,22))
	final_array+=0

	for i in range(11):
		for j in range(22):
			if i>=1 and i<=10 and j>=12 and j<=21:
				final_array[i][j]=pi[i-1][j-12]
	return final_array



class Agent(object):
	def __init__(self):
		self.episode_map=[]
		self.dealer_num=None
		self.final_player_sum=None
		self.main_list=[]
		self.key=-1
		self.key_link=-1
		self.policy=np.zeros(400)
		self.policy+=-1
		self.policy_with_ace=np.zeros((10,10))
		self.policy_with_ace+=-2
		self.policy_without_ace=np.zeros((10,10))
		self.policy_without_ace+=-2

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

		if key_link==None:
			self.key_link+=1
			key_link=self.key_link

		dict={	
				"key":self.key,
				"key-link":key_link,
				"Ace":ace,
				"Player_sum":player_sum,
				"Dealer_sum":dealer_num,
				"Action":action,
				"Action_Value":0,
				"Count":1,
				}

		self.main_list.append(dict)

		return dict["key"]

	def ace_state(self,player_sum):
		if player_sum+11<=21:
			ace_status=1 														
		else:
			ace_status=0

		return ace_status

	def episode(self,key):
		self.episode_map.append(key)


	def compute_action_value(self,reward):

		for i in range(len(self.episode_map)):
			key=self.episode_map[i]
			for j in range(len(self.main_list)):
				dict=self.main_list[j]
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
		
			for i in range(len(self.main_list)):
				dict=self.main_list[i]

				if dict["key"]==key:
					key_link=dict["key-link"]
					top=dict["Action_Value"]

					for j in range(len(self.main_list)):
						dict_1=self.main_list[j]
						if dict_1["key-link"]==key_link and dict_1["key"]!=key:

							if top<dict_1["Action_Value"]:
								action_val=(dict_1["Action_Value"])
								action=dict_1["Action"]
								top=action_val

					if action=="hit":
						self.policy[key_link]=0
					if action=="stick":
						self.policy[key_link]=1

			

	def classify_policy(self):

		for key_link in range(len(self.policy)):
			for i in range(len(self.main_list)):
				dict=self.main_list[i]
				if dict["key-link"]==key_link:
					player_sum=dict["Player_sum"]
					dealer_num=dict["Dealer_sum"]
					##print("Player_sum: " + str(player_sum))
					##print("Dealer_sum: " + str(dealer_num))
					if dict["Ace"]==1:
						self.policy_with_ace[dealer_num-1][player_sum-12]=self.policy[key_link]
					if dict["Ace"]==0:
						self.policy_without_ace[dealer_num-1][player_sum-12]=self.policy[key_link]


	def action(self,dealer_num,player_sum,ace):
		key_link=None
		response=None

		for i in range(len(self.main_list)):
			dict=self.main_list[i]
			if (dict["Ace"]==ace and dict["Player_sum"]==player_sum and dict["Dealer_sum"]==dealer_num ):
				key_link=dict["key-link"]
				break

		if key_link!=None:
			if self.policy[key_link]!=-1:
				if self.policy[key_link]==0:
					response="hit"
				if self.policy[key_link]==1:
					response="stick"

		if key_link==None or self.policy[key_link]==-1:
			response=None
			if player_sum<20:
				response="hit"
			if player_sum==20 or player_sum==21:
				response="stick"

		if key_link !=None:
			pass
			
		return response


	def reset_episode(self):
		self.episode_map*=0


	#def print(self):
		#print("value_estimates:" + str(self.value_estimates))
		#print("counts: "+ str(self.num_count))



class Environment(object):
	def __init__(self, agents, num_episodes):

		self.agents = agents
		self.player_sum=0
		self.dealer_num=0
		self.response=None
		self.ace_status_main=None
		self.num_episodes = num_episodes


	def dealer_response(self):
		while self.dealer_num<17:
			self.dealer_num+=np.random.randint(low=1,high=11)

		return self.dealer_num


	def reward_function(self,dealer_num,player_sum):
		reward=None
		if player_sum<=21 and player_sum>dealer_num:
			reward=1
		if player_sum<=21 and player_sum==dealer_num:
			reward=0
		if player_sum<=21 and dealer_num<=21 and player_sum<dealer_num:
			reward=-1
		if player_sum>21:
			reward=-1
		if player_sum<=21 and dealer_num>21:
			reward=1

		return reward

	def call_cards(self):

		while self.player_sum<=21:

			if self.response=="hit":

				self.player_sum+=np.random.randint(low=1,high=11)

				if self.player_sum<=21:

					self.response=self.agents.action(self.dealer_num,self.player_sum,self.ace_status_main)
					key=self.agents.add_state(self.dealer_num,self.player_sum,self.ace_status_main,self.response)
					self.agents.episode(key)

			if self.response=="stick":
				self.agents.final_player_sum=self.player_sum
				break

	def play(self):

		ace_status_main=0
		start=0
		response="abc"

		for num_episodes in tqdm(range(self.num_episodes)):
			response=None
			#print("New epsiode started")
		
			self.agents.reset_episode()
			self.dealer_num=np.random.randint(low=1,high=11)

			get_ace=np.random.randint(low=0,high=2)
			if get_ace==0:
				self.player_sum=np.random.randint(low=12,high=22)
				self.ace_status_main=0

			if get_ace==1:
				self.player_sum=11+np.random.randint(low=1,high=11)
				self.ace_status_main=1

			random_action=np.random.randint(low=0,high=2)
			if random_action==0:
				self.response="hit"
			else:
				self.response="stick"

			key=self.agents.add_state(self.dealer_num,self.player_sum,self.ace_status_main,self.response)
			self.agents.episode(key)

			self.call_cards()

			if self.player_sum>21:
				if self.ace_status_main==1:
					self.player_sum=self.player_sum-10
					self.ace_status_main=0
					self.response=self.agents.action(self.dealer_num,self.player_sum,self.ace_status_main)
					key=self.agents.add_state(self.dealer_num,self.player_sum,self.ace_status_main,self.response)
					self.agents.episode(key)

					self.call_cards()


				if self.ace_status_main==0:
					self.agents.final_player_sum=self.player_sum

			final_dealer_sum=self.dealer_response()
			reward=self.reward_function(final_dealer_sum,self.agents.final_player_sum)
			self.agents.compute_action_value(reward)
			self.agents.policy_update(self.agents.episode_map)


		self.agents.classify_policy()
		#print(self.agents.main_list)
		return self.agents.policy_with_ace,self.agents.policy_without_ace

################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution                 
	num_episodes= 1000000              # number of episodes

	agents = Agent()
	environment = Environment(agents=agents,num_episodes=num_episodes)

	 #Run Environment
	print("Running...")
	pi,pi_not = environment.play()
	print("Policy with Ace: ")
	print(pi)
	pi_final=accomdate_axis_change(pi)
	print("Policy without Ace: ")
	print(pi_not)
	pi_not_final=accomdate_axis_change(pi_not)
	print("Execution time: %s seconds" % (time.time() - start_time))
	

	fig, (ax0, ax1)=plt.subplots(2,1)


	c= ax0.pcolor(pi_final.T,edgecolors='k', linewidths=4)	
	ax0.set(xlim=(1,11),ylim=(12,22),xlabel='Dealers Card',ylabel='Player Sum',title='With Ace(Purple stands for "hit",Yellow stands for "stick")')


	c= ax1.pcolor(pi_not_final.T,edgecolors='k', linewidths=4)
	ax1.set(xlim=(1,11),ylim=(12,22),xlabel='Dealers Card',ylabel='Player Sum',title='Without Ace(Purple stands for "hit",Yellow stands for "stick")')
	
	fig.tight_layout()
	plt.show()



