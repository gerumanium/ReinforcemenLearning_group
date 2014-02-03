#n_armed_bandit.py
#
# n-armed bandit task chapter 2 Figure 2.1, 2.4
#
# 1. implemented with softmax function or epsilon-greedy function
# 2. implemented with online method for values
# 3. implemented with optimistic comparison
# 

import numpy as np
import matplotlib.pyplot as plt

#nBandit: the number of bandits
#nArm: the number of arms
#nPlay: the number of plyas (times we will pull a arm)
#sigma the standard deviation of the return from each of the arms

#Epsilon Greedy Method: x=qt, y=teps, z=_
def Greedy_method(Q, epsilon, numA):
	qt = Q[0];
	if np.random.rand() <= epsilon: #epsilon part
		arm = np.random.randint(0, numA);
	else: #greedy part
		#arm = np.where(max(qT[bi, :])==qT[bi,:])[0][0];
		arm = np.argmax(qt);
	return arm


#Softmax method: x=qt, y=teps
def Softmax_method(Q, epsilon):
	temperature = epsilon;
	qt  = Q[0];
	softe = np.exp(qt/temperature);

	#softmax function
	soft_max = softe / np.sum(softe);
	z = np.random.rand();
	
	#cumulative probability
	cum_prob = 0.0;
	for i in xrange(len(soft_max)):
		prob = soft_max[i];
		cum_prob += prob;
		if cum_prob > z:
			arm = i
			return arm
	return len(soft_max) - 1

#optimistic comparison
#x = optomistic, y = nArm, z = ei
def initial_velection(optimis, numA, te):
	qN = np.zeros((1, numA));
	#inital values set to zeros
	if optimis == 0:
		qT = np.zeros((1, numA));
	#initial values set to be optimistic
	else:
		if te == 0:
			qT = np.ones((1, numA)) * 5.0; #optomistic value
		else:
			qT = np.zeros((1, numA));
	return qT, qN


#main code
#nBandit: the number of simulation
#nArm: the number of Arms
#nPlay: the play times
#sigma: variance of rewards
#func_selection: choose either epsilon-method(func_selection=1) or softmax method(others)
#optimistic: set optimistically initial values(optimistic=1) or not(others)
def n_armed_testbed(nBandit=2000, nArm=10, nPlay=1000, sigma=1.0, func_selection=0, optimistic=0):
	#function = 1 for epsilon-greedy method else for softmax function

	if optimistic == 0 and func_selection == 0:
		#A set of epsilon for softmax function
		eps = [0.01, 0.1, 1];
	elif optimistic == 0 and func_selection == 1:
		#set of epsilon for greedy method
		eps = [0, 0.01, 0.1];
	else:
		#comparison between optimistic or nonoptimistic case
		eps = [0.0, 0.1];

	#the true reward from multiple gaussian distribution
	qTmean = np.random.multivariate_normal(np.zeros((nArm)), np.eye(nArm), (nBandit));
	
	#initial values for
	[row, column] = np.shape(qTmean);
	qT0 = np.zeros((row,column));

	#result containers
	average_reward = np.zeros((len(eps), nPlay));
	perOptAction = np.zeros((len(eps), nPlay));
	
	#make loops for each epsilon parameter
	for ei in xrange(len(eps)):
	
		#pick up a epsilon
		teps = eps[ei];
	
		#initialization of Action Values
		#make loops for each bandit
		Rewards = np.zeros((nBandit, nPlay));
		optAction = np.zeros((nBandit, nPlay));
	
		for bi in xrange(nBandit):

			#optimistic values
			#qT = np.zeros((1, nArm)); #initialization of values
			#qN = np.zeros((1, nArm)); #tracks of the number draws on each arm
			qT, qN = initial_velection(optimistic, nArm, ei);

			#make loops for each play
			for p in xrange(nPlay):
	
				#choose either epsilon-greedy or softmax
				#epsilon-greedy
				if func_selection == 1 :
					arm = Greedy_method(qT, teps, nArm);
				#choose softmax method
				else:
					arm = Softmax_method(qT, teps);
				
				#extract best arm choice
				best_arm = np.argmax(qTmean[bi, :]);
				
				#if selected arm is equivalent to best_arm, then count.
				if arm == best_arm:
					optAction[bi, p] = 1.0;
				
				#get the reward from drawing on that arm with qTmean + gaussian(mean=0,sigma=1)
				reward = qTmean[bi, arm] + sigma * np.random.normal(0,1);
				Rewards[bi, p] = reward;
				
				#update qN, qT
				#qT[0, arm] = qT[0, arm] + (reward - qT[0, arm])/(qN[0, arm] + 1);
				qT[0, arm] = qT[0, arm] + 0.1*(reward - qT[0, arm]);
 				#qN[0, arm] = qN[0, arm] + 1.0;
		
		#calculation of average action
		avg = np.mean(Rewards, 0);
 		average_reward[ei, :] = avg.T;

 		#calculation of optimal action
 		PercentOptAction = np.mean(optAction, 0);
	 	perOptAction[ei, :] = PercentOptAction.T;
	

	#plotting label w.r.t epsilon
	plot_label = "epsilon=";
	list_label = [];
	for i in eps:
		list_label.append(plot_label + str(i));

	#plotting for Average Reward
	x = np.linspace(0, nPlay, nPlay);
	for i in xrange(len(eps)):
		plt.plot(x, average_reward[i]);

	plt.xlabel('Plays');
	plt.ylabel('Average Rewards');
	plt.legend(list_label);
	plt.show()

	#plotting for Optimal Action
	for i in xrange(len(eps)):
		plt.plot(x, perOptAction[i]);
	
	plt.xlabel('Plays')
	plt.ylabel('Optimal Actions(%)')
	plt.legend(list_label);
	plt.show()


def main():
	n_armed_testbed();


if __name__ == "__main__":
	main()
