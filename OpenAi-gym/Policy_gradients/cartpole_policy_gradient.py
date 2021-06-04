import numpy as np 
import gym
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt 
from tqdm import tqdm

env = gym.make('CartPole-v0')

#CONSTANTS
MAX_EPISODES = 1000
GAMMA = 0.99
HIDDEN_LAYERS = 2

train_loss_results = []
class myModel(Model):

	def __init__(self, output_shape):
		super(myModel,self).__init__()
		self.layer1 = Dense(32, activation='relu', name='layer1', input_shape=(4,))
		self.layer2 = Dense(32, activation='relu', name='layer2')
		self.layer3 = Dense(32, activation='relu', name='layer3')
		self.layer4 = Dense(output_shape, activation='softmax', name='layer4')

	def call(self, input_data):
		input_data = tf.convert_to_tensor(input_data)
		input_data= tf.reshape(input_data, shape=(1,input_data.shape[0]))
		#input_data = tf.linalg.normalize(input_data)[0]

		output = self.layer1(input_data)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		return output

def get_discounted_reward(episodic_reward):
	discounted_reward = []
	discounted_future_reward = 0

	for i in range(len(episodic_reward)):
		discounted_future_reward = 0
		for j in range(len(episodic_reward) - i):
			discounted_future_reward += (GAMMA**(j))*episodic_reward[j]

		discounted_reward.append(discounted_future_reward)
	'''
	normalized_discounted_reward = []
	mean = np.mean(discounted_reward)
	std = np.std(discounted_reward)
	for i in range(len(discounted_reward)):
		discounted_reward[i] = (discounted_reward[i] - mean)/(std)
	'''
	#print(discounted_reward)
	return discounted_reward	

def get_action(policy, state, num_actions):
	state = tf.convert_to_tensor(state)
	probabilities = policy(state).numpy()[0]
	action = np.random.choice(num_actions, p=probabilities)

	return action, probabilities

def loss_function(prob, action, reward):
	selected_probs = tf.math.log((tf.reduce_sum(prob * tf.one_hot(action, num_actions),keepdims=[1])))
	cost = -tf.reduce_sum(reward * selected_probs)
	return cost

def update_policy(policy, states, actions, discounted_rewards):
	opt = tf.keras.optimizers.Adam(learning_rate=0.0001, 
									beta_1=0.8,
    								beta_2=0.999,
    								epsilon=1e-05,
    								amsgrad=True,)
	episodic_loss = []

	for state, reward, action in zip(states, discounted_rewards, actions):
		with tf.GradientTape() as tape:
			prob = policy(state)
			loss = loss_function(prob, action, reward)
			episodic_loss.append(loss)
			gradients = tape.gradient(loss, policy.trainable_variables)

		opt.apply_gradients(zip(gradients, policy.trainable_variables))

	train_loss_results.append(np.sum(episodic_loss))


if __name__ == '__main__':
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.n
	policy = myModel(num_actions)
	initial_state = env.reset()

	policy(initial_state)
	print(policy.summary())
	tf.keras.utils.plot_model(policy, to_file='my_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
	total_steps = []
	avg_episodic_reward = []

	for i in range(MAX_EPISODES):
		steps = 0
		#env.render()
		episodic_states = []
		#episodic_action_prob = []
		episodic_actions = []
		episodic_reward = []

		state = env.reset()
		done = False
		episodic_states.append(state)

		while not done:
			steps+=1
			action, prob = get_action(policy, state, num_actions)
			next_state, reward, done, _ = env.step(action)
			episodic_states.append(next_state)
			episodic_actions.append(action)
			#episodic_action_prob.append(prob)

			episodic_reward.append(reward)

			if done:
				discounted_reward = get_discounted_reward(episodic_reward)
				update_policy(policy, episodic_states, episodic_actions, discounted_reward)
				#print("Total number of steps are: " + str(np.sum(episodic_reward)))
				total_steps.append(np.sum(episodic_reward))
				avg_reward = np.mean(total_steps[-40:])
				print("EPISODE NUMBER: " + str(i))
				print("Episodic * {} * Avg Reward is ==> {}".format(np.sum(episodic_reward), avg_reward))
				avg_episodic_reward.append(avg_reward)
				print()
				#print(steps)

			else:
				state = next_state	

	plt.style.use('fivethirtyeight')
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	axes[0].set_ylabel("Policy Loss", fontsize=14)
	axes[0].set_xlabel("Episodes", fontsize=14)
	axes[0].plot(train_loss_results)

	axes[1].set_ylabel("Average Reward", fontsize=14)
	axes[1].set_xlabel("Episodes", fontsize=14)
	axes[1].plot(avg_episodic_reward)

	plt.show()
