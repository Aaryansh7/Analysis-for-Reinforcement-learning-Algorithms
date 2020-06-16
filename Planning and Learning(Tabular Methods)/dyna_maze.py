'''
###############################################################################################
	Dyna-Maze Example (Reinforcement Learning: An Introduction, Sutton, Barto)

	Program is about the Dyna Mze example in Planning and learning with tabuar methods 
	chapter, presented in the Reinforcement Learning: An Introduction book, Sutton, Barto, 
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
		self.action_val=np.zeros((54,4))
		self.model=[]
		self.step_size=step_size
		self.epsilon=epsilon
		self.goal_pos=[0,8]
		self.discount=0.95

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
		
		goal_pos=(self.goal_pos[0]*9)+ self.goal_pos[1]
		if cur_s==goal_pos:
			self.action_val[prev_s][prev_a]+=self.step_size*(reward - self.action_val[prev_s][prev_a])
		else:
			self.action_val[prev_s][prev_a]+=self.step_size*(reward + (self.discount*self.action_val[cur_s][cur_a]) - self.action_val[prev_s][prev_a])

	def model_update(self,reward,prev_s,prev_a,cur_s):
		flag=0
		for i in range(len(self.model)):
			if (self.model[i][0]==prev_s and self.model[i][1]==prev_a):
				flag=1
				break

		if flag!=1:
			self.model.append((prev_s,prev_a,reward,cur_s))


class Environment(object):
	def __init__(self,agents,num_episodes,planning_steps,runs):
		self.grid_map=np.zeros((6,9))

		self.grid_map[1][2]=1
		self.grid_map[2][2]=1
		self.grid_map[3][2]=1
		self.grid_map[4][5]=1
		self.grid_map[0][7]=1
		self.grid_map[1][7]=1
		self.grid_map[2][7]=1

		self.grid_walk=np.zeros((6,9))
		self.pos=[0,0]
		self.start_pos=[2,0]
		self.goal_pos=[0,8]
		self.num_episodes=num_episodes
		self.agents=agents
		self.planning_steps=planning_steps
		self.runs=runs

	def move_env(self,action,pos):
		fin_pos=pos.copy()

		if action=="left":
			fin_pos[1]-=1
			
		if action=="right":
			fin_pos[1]+=1

		if action=="down":
			fin_pos[0]+=1

		if action=="up":
			fin_pos[0]-=1

		if fin_pos[0]<0:
			fin_pos[0]=0
		if fin_pos[0]>5:
			fin_pos[0]=5
		if fin_pos[1]<0:
			fin_pos[1]=0
		if fin_pos[1]>8:
			fin_pos[1]=8

		if self.grid_map[int(fin_pos[0])][int(fin_pos[1])]==1:
			return pos

		return fin_pos

	def reward(self,new_pos):
		if new_pos==self.goal_pos:
			reward=1
		if new_pos!=self.goal_pos:
			reward=0
		return reward


	def play(self):
		cnt_epsiode=[]
		episode=[]

		for num_episodes in tqdm(range(self.num_episodes)):
			num_steps=0
			#print("New Episode..")
			for runs in range(self.runs): 
				pos=self.start_pos.copy()
				state=int((pos[0]*9)+(pos[1]))
				a,action=agents.action_policy(state)
				self.grid_walk[2][0]=1

				while pos!=self.goal_pos:
					#print("New Step")
					#print("position: " + str(pos))
					#print("Action: " + str(action))


					new_pos=self.move_env(action,pos)
					#print("New position: " + str(new_pos))
					new_state=int((new_pos[0]*9)+(new_pos[1]))
					new_a,new_action=agents.action_policy(new_state)

					reward=self.reward(new_pos)
					agents.model_update(reward,state,a,new_state)

					agents.compute(reward,state,a,new_state,new_a)

					for n in range(self.planning_steps):
						#print("loop for model plan")
						max_index=len(agents.model)
						random_index=np.random.randint(low=0,high=(max_index))

						random_pair=agents.model[random_index]
						b,c=agents.action_policy(random_pair[3])

						agents.compute(random_pair[2],random_pair[0],random_pair[1],random_pair[3],b)

					pos=new_pos
					state=new_state
					a=new_a
					action=new_action

					num_steps+=1

					if num_episodes==1:
						self.grid_walk[int(pos[0])][int(pos[1])]+=1

			num_steps=num_steps/(self.runs-1)


			cnt_epsiode.append(num_steps)
			episode.append(num_episodes)
			#print("Epsiode ends")

		return cnt_epsiode,episode,self.grid_walk

################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution
	num_episodes=50
	runs_per_episode=10

	agents = Agent(step_size=0.1,epsilon=0.1)
	environment = Environment(agents=agents,num_episodes=num_episodes,planning_steps=50,runs=runs_per_episode)

	# Run Environment
	print("Running...")
	g1,g2,g3= environment.play()
	print("Execution time: %s seconds" % (time.time() - start_time))

	
	#Graph 1 

	plt.title("Dyna-Q Perfomance(Planning Steps=0)")
	plt.plot(g2,g1)
	plt.ylabel('Steps')
	plt.xlabel('Epsiodes')
	plt.tight_layout()
	plt.show()

	#Graph 2
	fig, (ax0)=plt.subplots(1,1)
	c= ax0.pcolor(g3.T,edgecolors='k', linewidths=4)
	ax0.set(title='GridWalk(n=50)')
	fig.colorbar(c)	
	fig.tight_layout()
	plt.show()









