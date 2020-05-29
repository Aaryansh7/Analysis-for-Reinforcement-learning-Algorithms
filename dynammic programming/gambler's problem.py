'''
###############################################################################################
	Gambler's Problem (Reinforcement Learning: An Introduction, Sutton, Barto)

	Program is about hte gambler's problem in dynammic programming, presented
	in the Reinforcement Learning: An Introduction book, Sutton, Barto, 
###############################################################################################
'''
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class Agent(object):
	
	def __init__(self,nAct,Prob,theta,sweep):
		self.value_function=np.zeros(nAct)
		self.policy=np.zeros(nAct) 
		self.action=None
		self.reward=None
		self.Prob=Prob               #probability for heads
		self.theta=theta             #limit of delta in value iteration
		self.sweep=sweep             #no.of sweeps

	def reset(self):
		self.value_function*=0

	def __str__(self):
		
		return " Sweep = " +str(self.sweep)


	def q_greedify_policy(self, state):
		val_max=[]
		max=float("-inf")
		for act in range(state):
			exp_return=self.expectedReturn(state,act)
			if exp_return>max:
				max=exp_return
				val_max*=0
			if exp_return==max:
				val_max.append(act)
		
		index=np.random.randint(low=0,high=len(val_max))
		action=val_max[index]
		self.policy[state]=action



	def value_iteration(self):
		#while True:                                        ##I have written both the codes for sweeps as well as 
			#delta = 0                                        # by using delta/theta method.Because the book has graphs for sweeps
		for sweep in range(self.sweep):                         #hence it is not commented
			for s in range(1,len(self.value_function)):
				v = self.value_function[s]
				self.bellman_optimality_update(s)
				#delta = max(delta, abs(v - self.value_function[s]))
			#if delta < self.theta:
				#break

		for s in range(1,len(self.policy)):
			self.q_greedify_policy(s)

		V=self.value_function
		pi=self.policy
		return V, pi

	def bellman_optimality_update(self,s):
	
		V_update=[]
		for act in range(s):
			exp_return=self.expectedReturn(s, act)
			V_update.append(exp_return)
		self.value_function[s]=max(V_update)


	def expectedReturn(self,state, action):
		outcomes = [0, 1]
		exp_return1= None
		exp_return2=None
		Prob=self.Prob
		for index in outcomes:
			if (index == 0):
				nextState = state - action
				probs = 1 - Prob
				reward = 0
				exp_return1=probs*(reward+self.value_function[nextState])
			if (index ==1):
				nextState = state + action
				if nextState>=100 :
					reward=1
					probs=Prob
					exp_return2=probs*(reward)
				else:
					reward=0
					probs = Prob
					exp_return2=probs*(reward+self.value_function[nextState])
		return exp_return1+exp_return2


class Environment(object):

	def __init__(self, agents, iterations):

		self.agents = agents
		self.iterations = iterations

	def play(self):
		V=None
		pi=None
		main_V=np.zeros((len(self.agents),100))
		main_pi=np.zeros((len(self.agents),100))
		agntcnt=0

		for iIter in tqdm(range(self.iterations)):
			for agent in self.agents:

				V,pi=agent.value_iteration()
				main_V[agntcnt]=V
				main_pi[agntcnt]=pi
				agntcnt+=1

		return main_V.T,main_pi.T
				


################################################################
## MAIN ##
if __name__ == "__main__":
	start_time = time.time()    #store time to monitor execution
	nAct = 100                  
	iterations = 1                     # number of iteration

	agents = [Agent(nAct=nAct,Prob=0.4,theta=0.1,sweep=1),Agent(nAct=nAct,Prob=0.4,theta=0.1,sweep=2),Agent(nAct=nAct,Prob=0.4,theta=0.1,sweep=3),Agent(nAct=nAct,Prob=0.4,theta=0.1,sweep=32)]
	environment = Environment(agents=agents,iterations=iterations)

	# Run Environment
	print("Running...")
	V,pi = environment.play()
	print("Execution time: %s seconds" % (time.time() - start_time))


	#Graph1
	plt.title("Value-Function")
	plt.plot(V)
	plt.ylabel('Value Estimates')
	plt.xlabel('State')
	plt.legend(agents, loc=4)
	plt.show()

	#Graph 2
	plt.title("Policy")
	plt.plot(pi)
	#plt.ylim(0, 100)
	plt.ylabel('policy')
	plt.xlabel('state')
	plt.legend(agents, loc=4)
	plt.show()



