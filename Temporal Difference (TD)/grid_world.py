'''
###############################################################################################
	Grid World Exampe(Sarsa) (Reinforcement Learning: An Introduction, Sutton, Barto)

	Program is about the Grid World in Temporal Difference Learning  chapter, presented
	in the Reinforcement Learning: An Introduction book, Sutton, Barto, 
###############################################################################################
'''

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors

plt.style.use('fivethirtyeight')

class Agent(object):
	def __init__(self,step_size,epsilon):
		self.action_val=np.zeros((70,4))
		self.step_size=step_size
		self.epsilon=epsilon
		self.goal_pos=[3,7]

	def action_map(self,a):
		if a==0:
			return "left"
		elif a==1:
			return "right"
		elif a==2:
			return "down"
		elif a==3:
			return "up"

	def argmax_action_state(self,state):
		top=float("-inf")
		ties=[]
		for i in range(4):
			if self.action_val[state][i]>top:
				top=self.action_val[state][i]
				ties*=0
			if self.action_val[state][i]==top:
				ties.append(i)

		index=np.random.randint(low=0,high=(len(ties)))
		return ties[index]

	def action_policy(self,state):
		action=None
		randProb = np.random.random()  
		if randProb < self.epsilon:
			a = np.random.randint(low=0,high=4)
			action=self.action_map(a) 
		else:
			a=self.argmax_action_state(state)
			action=self.action_map(a)
		return a,action

	def compute(self,reward,prev_s,prev_a,cur_s,cur_a):
		goal_pos=(self.goal_pos[0]*10)+ self.goal_pos[1]
		if cur_s==goal_pos:
			self.action_val[prev_s][prev_a]+=self.step_size*(reward - self.action_val[prev_s][prev_a])
		else:
			self.action_val[prev_s][prev_a]+=self.step_size*(reward + self.action_val[cur_s][cur_a] - self.action_val[prev_s][prev_a])

class Environment(object):
	def __init__(self,agents,num_episodes):
		self.grid_map=np.zeros((7,10))
		for i in range(7):
			self.grid_map[i][3]=1
			self.grid_map[i][4]=1
			self.grid_map[i][5]=1
			self.grid_map[i][6]=2
			self.grid_map[i][7]=2
			self.grid_map[i][8]=1
		self.grid_walk=np.zeros((7,10))
		self.pos=[0,0]
		self.start_pos=[3,0]
		self.goal_pos=[3,7]
		self.num_episodes=num_episodes
		self.agents=agents

	def move_env(self,action,pos):
		fin_pos=pos
		wind=self.grid_map[int(pos[0])][int(pos[1])]
		#print("Wind Value: "+ str(wind) )

		if action=="left":
			fin_pos[0]-=wind
			fin_pos[1]-=1
			
		if action=="right":
			fin_pos[0]-=wind
			fin_pos[1]+=1

		if action=="down":
			fin_pos[0]-=wind-1

		if action=="up":
			fin_pos[0]-=wind+1

		#print(fin_pos[0])
		if fin_pos[0]<0:
			fin_pos[0]=0
		if fin_pos[0]>6:
			fin_pos[0]=6
		if fin_pos[1]<0:
			fin_pos[1]=0
		if fin_pos[1]>9:
			fin_pos[1]=9

		return fin_pos

	def reward(self,new_pos):
		if new_pos==self.goal_pos:
			reward=0
		if new_pos!=self.goal_pos:
			reward=-1
		return reward


	def play(self):
		cnt_epsiode=[]
		episode=[]
		count=0
		for num_episodes in tqdm(range(self.num_episodes)):
			#print("New Episode..")
			pos=self.start_pos.copy()
			state=int((pos[0]*10)+(pos[1]))
			a,action=agents.action_policy(state)
			self.grid_walk[3][0]=1

			while pos!=self.goal_pos:
				#print("New Step")
				#print("position: " + str(pos))
				#print("Action: " + str(action))


				new_pos=self.move_env(action,pos)
				#print("New position: " + str(new_pos))
				new_state=int((new_pos[0]*10)+(new_pos[1]))
				new_a,new_action=agents.action_policy(new_state)

				reward=self.reward(new_pos)

				agents.compute(reward,state,a,new_state,new_a)

				pos=new_pos
				state=new_state
				a=new_a
				action=new_action

				if num_episodes==self.num_episodes-1:
					self.grid_walk[int(pos[0])][int(pos[1])]+=1

				count+=1

			cnt_epsiode.append(count)
			episode.append(num_episodes)

		return cnt_epsiode,episode,self.grid_walk

################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution
	num_episodes=200

	agents = Agent(step_size=0.5,epsilon=0.1)
	environment = Environment(agents=agents,num_episodes=num_episodes)

	# Run Environment
	print("Running...")
	g1,g2,g3= environment.play()
	print("Execution time: %s seconds" % (time.time() - start_time))

	
	#Graph 1 

	plt.title("Sarsa Perfomance")
	plt.plot(g1,g2)
	plt.ylabel('Epsiodes')
	plt.xlabel('Steps')
	plt.tight_layout()
	plt.show()

	#Graph 2
	fig, (ax0)=plt.subplots(1,1)
	c= ax0.pcolor(g3.T,edgecolors='k', linewidths=4)
	ax0.set(title='Grid Walk')
	fig.colorbar(c)	
	fig.tight_layout()
	plt.show()









