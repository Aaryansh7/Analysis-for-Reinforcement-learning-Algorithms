'''
###############################################################################################
	Random Walk (Reinforcement Learning: An Introduction, Sutton, Barto)

	Program is about the Random-Walk in Temporal Difference Learning  chapter, presented
	in the Reinforcement Learning: An Introduction book, Sutton, Barto, 
###############################################################################################
'''

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

class Monte_Carlo_Agent(object):
	def __init__(self,step_size):
		self.value_estimates=np.zeros(5)
		self.value_estimates+=0.5
		self.true_value_estimates=np.zeros(5)
		for i in range(len(self.true_value_estimates)):
			self.true_value_estimates[i]=(i+1)/6
		self.step_size=step_size
		self.episode_map=[]
		self.rms=[]

	def __str__(self):
		
		return "Monte-carlo"


	def compute(self,reward,current_pos,last_pos):

		if self.episode_map[-1]==0 or self.episode_map[-1]==6:
			for i in range(len(self.episode_map)-1):
				state=self.episode_map[i]-1
				self.value_estimates[state]+=self.step_size*(reward-self.value_estimates[state])		

	def episode(self,current_pos):
		self.episode_map.append(current_pos)

	def rms_compute(self):
		err=0
		for i in range(5):
			err+=(self.value_estimates[i]-self.true_value_estimates[i])**2

		rms=(err/5)**0.5
		self.rms.append(rms)


	def reset(self):
		self.episode_map*=0

	def print(self):
		print("Monte-carlo")

class TD_Agent(object):
	def __init__(self,step_size):
		self.value_estimates=np.zeros(5)
		self.value_estimates+=0.5
		self.true_value_estimates=np.zeros(5)
		for i in range(len(self.true_value_estimates)):
			self.true_value_estimates[i]=(i+1)/6
		self.step_size=step_size
		self.episode_map=[]
		self.rms=[]

	def __str__(self):
		
		return "TD"

	def compute(self,reward,current_pos,last_pos):
		if current_pos==0 or current_pos==6:
			self.value_estimates[last_pos-1]+=self.step_size*(reward-self.value_estimates[last_pos-1])
		else:
			self.value_estimates[last_pos-1]+=self.step_size*(reward+self.value_estimates[current_pos-1]-self.value_estimates[last_pos-1])		

	def episode(self,current_pos):
		self.episode_map.append(current_pos)

	def rms_compute(self):
		err=0
		for i in range(5):
			err+=(self.value_estimates[i]-self.true_value_estimates[i])**2

		rms=(err/5)**0.5
		self.rms.append(rms)

	def reset(self):
		self.episode_map*=0
	def print(self):
		print("TD")



class Environment(object):
	def __init__(self,agents,num_episodes):
		self.true_value_estimates=np.zeros(5)
		for i in range(len(self.true_value_estimates)):
			self.true_value_estimates[i]=(i+1)/6
		self.action_space=np.zeros(7)
		self.present_state=3
		self.last_state=3
		self.num_episodes=num_episodes
		self.agents=agents

	def action(self):
		action=np.random.randint(low=0,high=2)
		if action==0:
			return "Left"
		if action==1:
			return "Right"

	def reward_func(self,current_pos):
		if current_pos==6:
			reward=+1
		else:
			reward=0
		return reward

	def play(self):
		scoreArr = np.zeros((self.num_episodes, len(self.agents)))
		for num_episodes in tqdm(range(self.num_episodes)):
			self.present_state=3
			self.last_state=3
			for agent in self.agents:
				agent.reset()

			while self.present_state>0 and self.present_state<6:
				action=self.action()
				if action=="Left":
					self.present_state-=1
				if action=="Right":
					self.present_state+=1

				if self.present_state==0 or self.present_state==6:
					break

				reward=self.reward_func(self.present_state)

				for agents in self.agents:
					agents.episode(self.last_state)
					agents.compute(reward,self.present_state,self.last_state)
					

				self.last_state=self.present_state



			if self.present_state==0 or self.present_state==6:

				reward=self.reward_func(self.present_state)
				for agents in self.agents:
					agents.episode(self.present_state)
					agents.compute(reward,self.present_state,self.last_state)

			agntCnt=0
			for agents in self.agents:
				agents.rms_compute()
				scoreArr[num_episodes][agntCnt]=agents.rms[num_episodes]
				agntCnt+=1

		main_list=[]
		list1=self.true_value_estimates.tolist()
		main_list.append(list1)
		cnt=1
		for agents in self.agents:
			main_list.append(agents.value_estimates.tolist())
			cnt+=1

		return scoreArr,main_list

################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution
	num_episodes=100

	agents = [Monte_Carlo_Agent(step_size=0.01),TD_Agent(step_size=0.15)]
	environment = Environment(agents=agents,num_episodes=num_episodes)

	# Run Environment
	print("Running...")
	g1,g2= environment.play()
	print("Execution time: %s seconds" % (time.time() - start_time))


	#Graph 1 
	plt.title("Empirical RMS Erros avg over states")
	plt.plot(g1)
	plt.ylabel('Rms-Error')
	plt.xlabel('Episodes')
	plt.legend(agents, loc=4)
	plt.tight_layout()
	plt.show()

	
	#Graph 2 

	plt.title("Value Estimates")
	plt.plot(g2[0],label='True-value estimate',marker='o')
	#print("True-value Estimates: " + str(g2[0]))
	plt.plot(g2[1],label='Monte-carlo',marker='o')
	#print("Monte-carlo Estimates: " + str(g2[1]))
	plt.plot(g2[2],label='TD',marker='o')
	#print("TD: " + str(g2[2]))
	plt.ylabel('Value Estimates')
	plt.xlabel('States')
	plt.legend()
	plt.tight_layout()
	plt.show()
	












