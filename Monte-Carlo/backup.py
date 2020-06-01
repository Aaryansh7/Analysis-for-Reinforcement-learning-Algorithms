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
		##print("response value in add_state func(1):" +str(action))

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
		##print("response value in add_state func:" +str(action))

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
			'''
			top=float("-inf")
			
			for j in range(len(self.main_list)):
				dict=self.main_list[j]
				if dict["key"]==key:
					key_link=dict["key-link"]
					if top<dict["Action_Value"]:
						action_val=(dict["Action_Value"])
						action=dict["Action"]
						top=action_val
			'''
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

		##print("Key link is :" + str(key_link))
		if key_link !=None:
			pass
			##print("Policy action is: "  + str(self.policy[key_link]))
		##print("Response by action func is: "+  str(response))
		return response


	def reset_episode(self):
		self.episode_map*=0


	#def print(self):
		#print("value_estimates:" + str(self.value_estimates))
		#print("counts: "+ str(self.num_count))



class Environment(object):
	def __init__(self, agents, num_episodes):

		self.agents = agents
		#self.agents.reset()
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
			reward=-1
		if player_sum>21:
			reward=-1
		if player_sum<=21 and dealer_num>21:
			reward=1

		return reward


	def play(self):

		ace_status_main=0
		start=0
		response="abc"

		for num_episodes in tqdm(range(self.num_episodes)):
			response=None
			begin=0
			#print("New epsiode started")
		
			self.agents.reset_episode()
			dealer_num=np.random.randint(low=1,high=11)
			player_sum=np.random.randint(low=12,high=22)

			#print("Initial Player Sum: " + str(player_sum))
			#print("Initial Dealer Sum: " + str(dealer_num))

			random_action=np.random.randint(low=0,high=2)
			if random_action==0:
				response="hit"
			else:
				response="stick"

			#print("Random action selected: " + str(response))
			#print("State added...")

			key=self.agents.add_state(dealer_num,player_sum,ace_status_main,response)
			self.agents.episode(key)

			#print("Main list is: " + str(self.agents.main_list))
			#print("Epsiode Keys are: "+ str(self.agents.episode_map))

			#print("Entering While Loop")
			#print("Value of response before While loop: " + str(response))
			while player_sum<=21:

				#print("New iteration in  While Loop, Response is: " + str(response))

				if response=="hit":
					ace_status=np.random.randint(low=0,high=2)

					if ace_status==0:
						player_sum+=np.random.randint(low=1,high=11)
						ace_status_main=0

					if ace_status==1:
						ace_status_main=self.agents.ace_state(player_sum)
						if ace_status_main==1:
							player_sum+=11
						else:
							player_sum+=1


					if player_sum<=21:
						#print("New Player Sum is: "+ str(player_sum))
						#print("Status of Ace is" + str(ace_status_main))

						response=self.agents.action(dealer_num,player_sum,ace_status_main)
						#print("Reponse Generated is: " + str(response))
						key=self.agents.add_state(dealer_num,player_sum,ace_status_main,response)
						self.agents.episode(key)

						#print("Main list is: " + str(self.agents.main_list))
						#print("Epsiode Keys are: "+ str(self.agents.episode_map))

				if response=="stick":
					#print("Sticking at :" + str(player_sum))
					self.agents.final_player_sum=player_sum
					break

					##print("Response is Hit,ace status is: " + str(ace_status_main))

				##print("Response is Hit,ace status is: " + str(ace_status_main))
				#response=self.agents.action(dealer_num,player_sum,ace_status_main)
				'''
				if player_sum<=21:
					key=self.agents.add_state(dealer_num,player_sum,ace_status_main,response)
					self.agents.episode(key)

				##print("state added to epsiode")

				if response=="stick":
					self.agents.final_player_sum=player_sum
					break
				'''
			if player_sum>21:
				self.agents.final_player_sum=player_sum

			final_dealer_sum=self.dealer_response(dealer_num)
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
	num_episodes= 1000000               # number of episodes

	agents = Agent()
	environment = Environment(agents=agents,num_episodes=num_episodes)

	 #Run Environment
	print("Running...")
	pi,pi_not = environment.play()
	#print("Policy with Ace: ")
	#print(pi)
	print("Policy without Ace: ")
	print(pi_not)
	print("Execution time: %s seconds" % (time.time() - start_time))
	'''
	
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
	'''
