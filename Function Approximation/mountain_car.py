import gym
import numpy as np 
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
import tiles3 as tc

env=gym.make('MountainCar-v0')

done=False
count_episode=0


########################################

class MountainCarTileCoder:
	def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):

		self.iht = tc.IHT(iht_size)
		self.num_tilings = num_tilings
		self.num_tiles = num_tiles
	
	def get_tiles(self, position, velocity):

		POSITION_MIN=-1.2
		POSITION_MAX=0.6
		VELOCITY_MIN=-0.07
		VELOCITY_MAX=0.07
 
		position_scale = self.num_tiles /(POSITION_MAX-POSITION_MIN)
		velocity_scale=self.num_tiles/(VELOCITY_MAX-VELOCITY_MIN)

		tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, velocity * velocity_scale])
		
		return np.array(tiles)

#####################################################

class SarsaAgent:

	def __init__(self):
		self.last_action = None
		self.last_state = None
		self.epsilon = 0.1
		self.gamma = 1
		self.iht_size = 4096
		self.w = None
		self.num_tilings = 8
		self.alpha = 0.5/(self.num_tilings)
		self.num_tiles = 8
		self.mctc = None
		self.initial_weights = 0
		self.num_actions = 3
		self.previous_tiles = None

		self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
		self.tc = MountainCarTileCoder(iht_size=self.iht_size, num_tilings=self.num_tilings, num_tiles=self.num_tiles)

	def select_action(self,tiles):
		action_values = []
		chosen_action = None

		max_val=float("-inf")
		top=None
		for i in range(self.num_actions):
			action_num=self.w[i][tiles]
			val=np.sum(action_num)
			action_values.append(val)
			if val>max_val:
				top=i
				max_val=val

		epsilon=np.random.random(size=1)
		if epsilon<self.epsilon:
			chosen_action=np.random.randint(low=0,high=self.num_actions)
		else:
			chosen_action=top
		
		return chosen_action, action_values[chosen_action]

	def agent_start(self, state):

		position = state[0]
		velocity = state[1]

		active_tiles=self.tc.get_tiles(position, velocity)
		current_action,current_value=self.select_action(active_tiles)
		
		self.last_action = current_action
		self.previous_tiles = np.copy(active_tiles)

		return self.last_action

	def agent_step(self, reward, state):

		position = state[0]
		velocity = state[1]
		active_tiles=self.tc.get_tiles(position,velocity)
		current_action,current_value=self.select_action(active_tiles)
		previous_value = np.sum(self.w[self.last_action][self.previous_tiles])
		
		
		self.w[self.last_action][self.previous_tiles]+=self.alpha*(reward+self.gamma*current_value-previous_value)
		
		self.last_action = current_action
		self.previous_tiles = np.copy(active_tiles)
		return self.last_action

	def agent_end(self, reward):

		previous_value = np.sum(self.w[self.last_action][self.previous_tiles])
		self.w[self.last_action][self.previous_tiles]+=self.alpha*(reward-previous_value)

##################################################
if __name__ == "__main__":
	agent = SarsaAgent()
	num_runs=50
	main_count=[]
	run_count=[]

	for run in range(num_runs):
		observation=env.reset()
		done=False
		action = agent.agent_start(observation)
		count_episode=0
		while not done:
			env.render()
			observation,reward,done,_=env.step(action)
			#print("Reward" + str(reward))
			#print("observation" + str(observation))

			action=agent.agent_step(reward,observation)
			count_episode+=1

			if done:
				print("Complete,Final Reward:" + str(reward))
				print("Complete,Final Observation:" + str(observation))
				print("Episode Count:" + str(count_episode))
				main_count.append(count_episode)
				break
		run_count.append(run)
	print("Count for each epsiode: " + str(main_count))

	#Graph 1 	
	plt.title("MountainCar-v0")
	plt.plot(run_count,main_count)
	plt.ylabel('Number of episodes')
	plt.xlabel('runs')
	plt.legend()
	plt.tight_layout()
	plt.show()