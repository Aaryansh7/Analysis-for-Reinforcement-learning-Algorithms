import numpy as np 
import gym
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt 

env = gym.make('CartPole-v0')

#CONSTANTS
MAX_EPISODES = 400
GAMMA = 0.99

train_policy_loss_results = []
step_policy_loss = []
step_q_loss= []
train_q_function_loss_results = []

class Actor(Model):

	def __init__(self, output_shape):
		super(Actor,self).__init__()
		self.layer1 = Dense(32, activation='relu', name='layer1')
		self.layer2 = Dense(32, activation='relu', name='layer2')
		self.layer3 = Dense(32, activation='relu', name='layer3')
		self.layer4 = Dense(output_shape, activation='softmax', name='layer4')

	def call(self, input_data):
		input_data = tf.convert_to_tensor(input_data)
		input_data= tf.reshape(input_data, shape=(1,input_data.shape[0]))

		output = self.layer1(input_data)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		return output

class Critic(Model):

	def __init__(self):
		super(Critic,self).__init__()
		self.layer1 = Dense(32, activation='relu', name='layer1')
		self.layer2 = Dense(32, activation='relu', name='layer2')
		self.layer3 = Dense(32, activation='relu', name='layer3')
		self.layer4 = Dense(1, activation='relu', name='layer4')

	def call(self, state):
		input_data = tf.convert_to_tensor(state)
		input_data= tf.reshape(input_data, shape=(1,input_data.shape[0]))

		output = self.layer1(input_data)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		return output

def get_action(policy, state, num_actions):
	state = tf.convert_to_tensor(state)
	probabilities = policy(state).numpy()[0]
	action = np.random.choice(num_actions, p=probabilities)

	return action

def compute_td_error(Q, next_state, next_action, state, action, reward):
	q_next_value = Q(next_state)
	q_value = Q(state)

	td_error = reward + GAMMA*q_next_value - q_value
	return td_error


def loss_function_policy(prob, action, q_value):
	selected_probs = tf.math.log((tf.reduce_sum(prob * tf.one_hot(action, num_actions),keepdims=[1])))
	cost = -tf.reduce_sum(q_value * selected_probs)
	return cost

def update_policy(policy, Q, state, action):
	opt = tf.keras.optimizers.Adam(learning_rate=0.0001, 
									beta_1=0.8,
    								beta_2=0.999,
    								epsilon=1e-05,
    								amsgrad=True,)
	step_loss = []
	with tf.GradientTape() as tape:
		prob = policy(state)
		q_value = Q(state)
		loss = loss_function_policy(prob, action, q_value)
		step_loss.append(loss)
		gradients = tape.gradient(loss, policy.trainable_variables)

	opt.apply_gradients(zip(gradients, policy.trainable_variables))
	step_policy_loss.append(np.sum(step_loss))
	


def update_Q(Q,td_error, state, action):
	# Optimizer
	opt = tf.keras.optimizers.Adam(learning_rate=0.0003, 
									beta_1=0.9,
    								beta_2=0.999,
    								epsilon=1e-07,
    								amsgrad=True,)

	step_loss = []
	with tf.GradientTape() as tape:
		q_value = Q(state)
		loss = -tf.reduce_sum(q_value * td_error)
		step_loss.append(loss)
		gradients = tape.gradient(loss, Q.trainable_variables)

	opt.apply_gradients(zip(gradients, Q.trainable_variables))
	step_q_loss.append(np.sum(step_loss))
	


if __name__ == '__main__':
	num_states = env.observation_space.shape[0] 
	num_actions = env.action_space.n

	policy = Actor(num_actions) 
	initial_state = env.reset()
	Q = Critic()

	policy(initial_state) # policy network intiliased
	print(" Actor-Network: " + str(policy.summary()))

	Q(initial_state) # Value function network initialised
	print(" Critic-Network: " + str(Q.summary()))
	
	total_steps = []
	avg_episodic_reward = []

	for i in range(MAX_EPISODES):  # new episode started
		steps = 0
		#env.render()
		episodic_states = []
		episodic_actions = []
		episodic_reward = []
		step_policy_loss = []
		step_q_loss =[]

		state = env.reset()        # state set to initial
		action = get_action(policy, state, num_actions) # action sampled  
		done = False
		episodic_states.append(state)
		episodic_actions.append(action) # (s,a) found out

		while not done:
			steps+=1		

			# step in environment (R, s',a' ) sampled.	
			next_state, reward, done, _ = env.step(action)  
			next_action = get_action(policy, next_state, num_actions)  

			episodic_states.append(next_state)
			episodic_actions.append(next_action)
			episodic_reward.append(reward)

			update_policy(policy, Q, state, action) # weights for policy updated
			td_error = compute_td_error(Q, next_state, next_action, state, action, reward) # td error computed.
			update_Q(Q,td_error, state, action) # Value-functino updated

			state = next_state	
			action = next_action

			if done:
				total_steps.append(np.sum(episodic_reward))
				avg_reward = np.mean(total_steps[-40:])
				train_q_function_loss_results.append(np.mean(step_q_loss))
				train_policy_loss_results.append(np.mean(step_policy_loss))
				print("EPISODE NUMBER: " + str(i))
				print("Episodic * {} * Avg Reward is ==> {}".format(np.sum(episodic_reward), avg_reward))
				avg_episodic_reward.append(avg_reward)
				print()


	plt.style.use('fivethirtyeight')
	
	fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
	axes[0].set_ylabel("Policy_Loss", fontsize=14)
	axes[0].set_xlabel("Episodes", fontsize=14)
	axes[0].plot(train_policy_loss_results)

	axes[1].set_ylabel("Average Reward", fontsize=14)
	axes[1].set_xlabel("Episodes", fontsize=14)
	axes[1].plot(avg_episodic_reward)


	axes[2].set_ylabel("Q_Loss", fontsize=14)
	axes[2].set_xlabel("Episodes", fontsize=14)
	axes[2].plot(train_q_function_loss_results)

	plt.show()